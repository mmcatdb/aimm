"""
Usage:
    # Single query via argument
    python estimate_latency.py --query 'MATCH (n:Customer) RETURN count(n)'

    # Multiple queries from a file (one query per line or separated by semicolons)
    python estimate_latency.py --file queries.txt
"""
import torch
import pickle
import argparse
import sys

from common.config import Config
from plan_extractor import Neo4j, PlanExtractor
from feature_extractor import FeatureExtractor
from plan_structured_network import PlanStructuredNetwork
from neural_units import create_neural_unit


class LatencyEstimator:
    """
    Estimates query latency using a trained model without executing queries.
    Uses EXPLAIN to get the query plan and neural network for prediction.
    """

    def __init__(self, neo4j: Neo4j, checkpoint_path: str, feature_extractor_path: str | None = None, device: str = 'cpu', num_layers: int = 5, hidden_dim: int = 128):
        self.extractor = PlanExtractor(neo4j)
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Derive feature extractor path if not provided
        if feature_extractor_path is None:
            feature_extractor_path = checkpoint_path.replace('.pt', '_feature_extractor.pkl')

        # Load feature extractor
        try:
            with open(feature_extractor_path, 'rb') as f:
                self.feature_extractor = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Feature extractor not found at {feature_extractor_path}. '
                f'Please ensure the feature extractor file exists or specify --feature-extractor path.'
            )

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Model checkpoint not found at {checkpoint_path}. '
                f'Please ensure the checkpoint file exists or specify --checkpoint path.'
            )

        # Create model
        self.model = PlanStructuredNetwork(
            feature_extractor=self.feature_extractor,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            data_vec_dim=32
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
                    operator_feature_dim = input_dim - (data_vec_dim * num_children)

                    unit = create_neural_unit(
                        operator_type=operator_type,
                        input_dim=operator_feature_dim,
                        num_children=num_children,
                        data_vec_dim=data_vec_dim,
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers
                    )
                    key = self.model._get_unit_key(operator_type, num_children)
                    self.model.units[key] = unit
                    self.model.operator_types.add(operator_type)

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        # Store model info
        self.epoch = checkpoint.get('epoch', 'unknown')

    def close(self):
        """Close database connection."""
        if self.extractor:
            self.extractor.close()

    def estimate(self, query: str) -> tuple[float, dict]:
        """
        Estimate query latency without executing the query.

        Args:
            query: Cypher query string

        Returns:
            Tuple of (estimated_latency_seconds, query_plan)
        """
        # Get plan using EXPLAIN
        plan = self.extractor.get_plan(query)

        # Predict latency using the trained model
        with torch.no_grad():
            predicted_latency = self.model.forward(plan)

        return predicted_latency.item(), plan

    def estimate_batch(self, queries: list[str]) -> list[tuple[str, float, dict]]:
        """
        Estimate latency for multiple queries.

        Args:
            queries: List of Cypher query strings

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
    parser = argparse.ArgumentParser()

    # Query input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--query', '-q', type=str, help='Single Cypher query to estimate')
    input_group.add_argument('--file', '-f', type=str, help='File containing queries (one per line or semicolon-separated)')

    # Model options
    parser.add_argument('--checkpoint', '-c', type=str, default='data/neo4j_qpp_checkpoint.pt', help='Path to model checkpoint (default: data/neo4j_qpp_checkpoint.pt)')
    parser.add_argument('--feature-extractor', type=str, default=None, help='Path to feature extractor pickle (default: derived from checkpoint path)')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension of neural units (must match trained model)')
    parser.add_argument('--num-layers', type=int, default=5, help='Number of layers per neural unit (must match trained model)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference')

    # Output options
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output including query plans')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--quiet', action='store_true', help='Only output the estimated latency value(s)')

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

        config = Config.load()
        neo4j = Neo4j(config.neo4j)
        estimator = LatencyEstimator(
            neo4j=neo4j,
            checkpoint_path=args.checkpoint,
            feature_extractor_path=args.feature_extractor,
            device=args.device,
            num_layers=args.num_layers,
            hidden_dim=args.hidden_dim
        )
        if not args.quiet:
            print(f'Model loaded (trained for {estimator.epoch} epochs, '
                  f'{estimator.model.count_parameters():,} parameters)\n', file=sys.stderr)
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
                    print(f'Error: {plan["error"]}')
                    sys.exit(1)

                print(f'Query: {query.strip()}')
                print(f'Estimated latency: {format_latency(latency)}')

                if args.verbose:
                    print(f'\nQuery Plan:')
                    print(f'  Root operator: {plan.get("operatorType", "Unknown")}')
                    print(f'  Estimated rows: {plan.get("args", {}).get("EstimatedRows", "N/A")}')

            else:
                # Multiple queries - table format
                print(f'{"#":<4} {"Query":<60} {"Estimated Latency":<20}')
                print('-' * 84)

                for i, (query, latency, plan) in enumerate(results, 1):
                    query_display = truncate_query(query)
                    if 'error' in plan:
                        latency_str = f'ERROR: {plan["error"][:30]}'
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
