"""
Usage:
    # Single query via argument
    python estimate_latency.py --query 'SELECT COUNT(*) FROM lineitem'

    # Multiple queries from a file (one query per line or separated by semicolons)
    python estimate_latency.py --file queries.txt
"""
import torch
import argparse
import sys

from config import DatabaseConfig
from feature_extractor import FeatureExtractor
from plan_structured_network import PlanStructuredNetwork
from neural_units import GenericUnit


def create_neural_unit(operator_type: str, input_dim: int, num_children: int,
                       data_vec_dim: int, hidden_dim: int, num_layers: int):
    """Create a neural unit for a given operator type."""
    return GenericUnit(
        input_dim=input_dim,
        num_children=num_children,
        data_vec_dim=data_vec_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )


class LatencyEstimator:
    """
    Estimates query latency using a trained model without executing queries.
    Uses EXPLAIN to get the query plan and neural network for prediction.
    """

    def __init__(self, checkpoint_path: str,
                 device: str = 'cpu', num_layers: int = None, hidden_dim: int = None,
                 config_path: str = 'config.yaml'):
        self.device = device

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f'Model checkpoint not found at {checkpoint_path}. Please ensure the checkpoint file exists or specify --checkpoint path.')

        # Load feature extractor from checkpoint
        if 'feature_extractor' not in checkpoint:
            raise ValueError(f'Checkpoint does not contain feature_extractor. Please ensure you are using a checkpoint saved by the training script.')

        self.feature_extractor = checkpoint['feature_extractor']

        # Get model config from checkpoint, with argument values as fallback
        saved_config = checkpoint.get('config', {})
        self.hidden_dim = saved_config.get('hidden_dim') or hidden_dim or 128
        self.num_layers = saved_config.get('num_layers') or num_layers or 5
        data_vec_dim = saved_config.get('data_vec_dim', 32)

        # Create model
        self.model = PlanStructuredNetwork(
            feature_extractor=self.feature_extractor,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            data_vec_dim=data_vec_dim
        )

        # Pre-create neural units based on the checkpoint's state dict keys
        state_dict = checkpoint['model_state_dict']

        # Extract unique unit keys from state dict
        unit_keys = set()
        for key in state_dict.keys():
            if key.startswith('units.'):
                parts = key.split('.')
                if len(parts) >= 2:
                    unit_keys.add(parts[1])

        # Create each unit with exact dimensions from checkpoint
        for unit_key in unit_keys:
            parts = unit_key.rsplit('_', 1)
            if len(parts) == 2:
                operator_type = parts[0]
                num_children = int(parts[1])

                # Get the exact input dimension from the checkpoint
                weight_key = f'units.{unit_key}.hidden_layers.0.weight'
                if weight_key in state_dict:
                    weight_shape = state_dict[weight_key].shape
                    input_dim = weight_shape[1]

                    data_vec_dim = 32
                    operator_feature_dim = input_dim - (num_children * (1 + data_vec_dim))

                    unit = create_neural_unit(
                        operator_type=operator_type,
                        input_dim=operator_feature_dim,
                        num_children=num_children,
                        data_vec_dim=data_vec_dim,
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers
                    )
                    # Use the same key format as PlanStructuredNetwork
                    key = f'{operator_type}_{num_children}'
                    self.model.units[key] = unit

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # Store model info
        self.epoch = checkpoint.get('epoch', 'unknown')

        # Connect to PostgreSQL
        self.config = DatabaseConfig(config_path)

    def close(self):
        """Close database connection (should be no persistent connection for PostgreSQL)."""
        pass

    def get_plan(self, query: str) -> dict:
        """
        Get the query plan using EXPLAIN without executing the query.

        Args:
            query: SQL query string

        Returns:
            Query plan dictionary
        """
        conn = self.config.get_connection()

        try:
            conn.autocommit = True
            with conn.cursor() as cursor:
                # Get the plan without execution (no ANALYZE)
                explain_query = f'EXPLAIN (FORMAT JSON, VERBOSE) {query}'
                cursor.execute(explain_query)
                result = cursor.fetchone()

                # Parse JSON plan - EXPLAIN returns [{'Plan': {...}}]
                plan_json = result[0][0]

                return plan_json['Plan']
        finally:
            conn.close()

    def estimate(self, query: str) -> tuple[float, dict]:
        """
        Estimate query latency without executing the query.

        Args:
            query: SQL query string

        Returns:
            Tuple of (estimated_latency_seconds, query_plan)
        """
        # Get plan using EXPLAIN
        plan = self.get_plan(query)

        # Predict latency using the trained model
        with torch.no_grad():
            predicted_latency = self.model.forward(plan)

        return predicted_latency.item(), plan

    def estimate_batch(self, queries: list[str]) -> list[tuple[str, float, dict]]:
        """
        Estimate latency for multiple queries.

        Args:
            queries: List of SQL query strings

        Returns:
            List of tuples (query, estimated_latency_seconds, plan)
        """
        results = []
        for query in queries:
            try:
                latency, plan = self.estimate(query)
                results.append((query, latency, plan))
            except Exception as e:
                results.append((query, None, {'error': str(e)}))
        return results


def parse_queries_from_file(filepath: str) -> list[str]:
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove comments
    lines = []
    for line in content.split('\n'):
        # Remove inline comments
        if '#' in line:
            line = line[:line.index('#')]
        if '--' in line:
            line = line[:line.index('--')]
        lines.append(line)

    content = '\n'.join(lines)

    # Split by semicolons if present, otherwise by double newlines
    if ';' in content:
        queries = [q.strip() for q in content.split(';')]
    else:
        # Try splitting by double newlines for multi-line queries
        queries = [q.strip() for q in content.split('\n\n')]

        # If that results in single lines, treat each non-empty line as a query
        if all('\n' not in q for q in queries if q):
            queries = [q.strip() for q in content.split('\n')]

    # Filter out empty queries
    queries = [q for q in queries if q.strip()]

    return queries


def format_latency(latency_seconds: float) -> str:
    """Format latency for human-readable output."""
    if latency_seconds is None:
        return 'ERROR'
    if latency_seconds < 0.001:
        return f'{latency_seconds * 1000000:.2f} µs'
    elif latency_seconds < 1:
        return f'{latency_seconds * 1000:.2f} ms'
    else:
        return f'{latency_seconds:.3f} s'


def truncate_query(query: str, max_length: int = 60) -> str:
    """Truncate query for display."""
    query = ' '.join(query.split())  # Normalize whitespace
    if len(query) > max_length:
        return query[:max_length - 3] + '...'
    return query


def main():
    parser = argparse.ArgumentParser(
        description='Estimate PostgreSQL query latency using a trained neural network model.'
    )

    # Query input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--query', '-q', type=str,
                            help='Single SQL query to estimate')
    input_group.add_argument('--file', '-f', type=str,
                            help='File containing queries (one per line or semicolon-separated)')

    parser.add_argument('--checkpoint', '-c', type=str,
                       default='qpp_net_model.pt',
                       help='Path to model checkpoint (default: qpp_net_model.pt)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Hidden dimension of neural units (default: auto-detect from checkpoint)')
    parser.add_argument('--num-layers', type=int, default=None,
                       help='Number of layers per neural unit (default: auto-detect from checkpoint)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to database config file (default: config.yaml)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output including query plans')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    parser.add_argument('--quiet', action='store_true',
                       help='Only output the estimated latency value(s)')

    args = parser.parse_args()

    # Collect queries
    if args.query:
        queries = [args.query]
    else:
        try:
            queries = parse_queries_from_file(args.file)
            if not queries:
                print(f'Error: No queries found in {args.file}', file=sys.stderr)
                sys.exit(1)
        except FileNotFoundError:
            print(f'Error: File not found: {args.file}', file=sys.stderr)
            sys.exit(1)

    # Initialize estimator
    try:
        if not args.quiet:
            print('Loading model...', file=sys.stderr)
        estimator = LatencyEstimator(
            checkpoint_path=args.checkpoint,
            device=args.device,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim,
            config_path=args.config
        )
        if not args.quiet:
            num_params = sum(p.numel() for p in estimator.model.parameters())
            print(f'Model loaded (trained for {estimator.epoch} epochs, {num_params:,} parameters)\n', file=sys.stderr)
    except FileNotFoundError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    try:
        # Estimate latencies
        results = estimator.estimate_batch(queries)

        # Output results
        if args.json:
            import json
            output = []
            for query, latency, plan in results:
                item = {
                    'query': query,
                    'estimated_latency_seconds': latency,
                    'estimated_latency_formatted': format_latency(latency) if latency else None
                }
                if args.verbose:
                    item['plan'] = plan
                if 'error' in plan:
                    item['error'] = plan['error']
                output.append(item)
            print(json.dumps(output, indent=2))

        elif args.quiet:
            for query, latency, plan in results:
                if latency is not None:
                    print(f'{latency:.6f}')
                else:
                    print('ERROR')

        else:
            # Standard output
            if len(results) == 1:
                query, latency, plan = results[0]
                if 'error' in plan:
                    print(f'Error: {plan['error']}')
                    sys.exit(1)

                print(f'Query: {query.strip()}')
                print(f'Estimated latency: {format_latency(latency)}')

                if args.verbose:
                    print(f'\nQuery Plan:')
                    print(f'  Root operator: {plan.get('Node Type', 'Unknown')}')
                    print(f'  Estimated rows: {plan.get('Plan Rows', 'N/A')}')
                    print(f'  Total cost: {plan.get('Total Cost', 'N/A')}')

            else:
                # Multiple queries - table format
                print(f'{'#':<4} {'Query':<60} {'Estimated Latency':<20}')
                print('-' * 84)

                for i, (query, latency, plan) in enumerate(results, 1):
                    query_display = truncate_query(query)
                    if 'error' in plan:
                        latency_str = f'ERROR: {plan['error'][:30]}'
                    else:
                        latency_str = format_latency(latency)
                    print(f'{i:<4} {query_display:<60} {latency_str:<20}')

                # Summary
                valid_latencies = [lat for _, lat, plan in results if lat is not None and 'error' not in plan]
                if valid_latencies:
                    print('-' * 84)
                    print(f'Total queries: {len(results)}')
                    print(f'Successful estimates: {len(valid_latencies)}')
                    print(f'Average estimated latency: {format_latency(sum(valid_latencies) / len(valid_latencies))}')

    finally:
        estimator.close()


if __name__ == '__main__':
    main()
