"""
Evaluation script for the trained Neo4j Query Performance Predictor.

This script:
1. Loads a trained model
2. Generates new test queries
3. Predicts their latency using EXPLAIN (no execution)
4. Measures actual execution time
5. Compares predictions to actual times
"""
import torch
import numpy as np
import time
import pickle
import argparse
from tabulate import tabulate

from datasets.database import Database, TestQuery
from datasets.tpch.neo4j import TpchNeo4j
from common.config import Config
from common.drivers import Neo4jDriver, cypher
from plan_extractor import PlanExtractor
from feature_extractor import FeatureExtractor
from plan_structured_network import PlanStructuredNetwork

class ModelEvaluator:
    """Evaluates a trained Neo4j QPP model."""

    def __init__(self, neo4j: Neo4jDriver, database: Database, checkpoint_path: str, feature_extractor_path: str, device: str = 'cpu', num_layers: int = 10, hidden_dim: int = 128):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            feature_extractor_path: Path to feature extractor pickle file
            device: 'cpu' or 'cuda'
            num_layers: Number of layers in neural units
            hidden_dim: Hidden dimension of neural units
        """
        self.extractor = PlanExtractor(neo4j, database)
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Load feature extractor
        print('Loading feature extractor...')
        with open(feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)

        # Load checkpoint first to get operator info
        print('Loading model checkpoint...')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Create model
        print('Creating model architecture...')
        self.model = PlanStructuredNetwork(
            feature_extractor=self.feature_extractor,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            data_vec_dim=32
        )

        # Pre-create neural units based on the checkpoint's state dict keys
        # This ensures all units exist before loading weights
        print('Initializing neural units from checkpoint...')
        state_dict = checkpoint['model_state_dict']

        # Extract unique unit keys from state dict
        unit_keys = set()
        for key in state_dict.keys():
            if key.startswith('units.'):
                # Extract the unit key (e.g., 'ProduceResults_1' from 'units.ProduceResults_1.hidden_layers.0.weight')
                parts = key.split('.')
                if len(parts) >= 2:
                    unit_keys.add(parts[1])

        # Create each unit by parsing the key and getting exact dimensions from checkpoint
        for unit_key in unit_keys:
            # Parse operator_type and num_children from key (e.g., 'ProduceResults_1')
            parts = unit_key.rsplit('_', 1)
            if len(parts) == 2:
                operator_type = parts[0]
                num_children = int(parts[1])

                # Get the exact input dimension from the checkpoint
                # The first hidden layer weight has shape [hidden_dim, input_dim]
                weight_key = f'units.{unit_key}.hidden_layers.0.weight'
                if weight_key in state_dict:
                    weight_shape = state_dict[weight_key].shape
                    input_dim = weight_shape[1]  # input_dim is the second dimension

                    # For operators with children, we need to figure out the operator feature dim
                    # input_dim = operator_feature_dim + (data_vec_dim * num_children)
                    data_vec_dim = 32
                    operator_feature_dim = input_dim - (data_vec_dim * num_children)

                    # Directly create the unit with the correct dimensions
                    # We bypass _ensure_unit_exists to create with exact dimensions
                    from neural_units import create_neural_unit
                    unit = create_neural_unit(
                        operator_type=operator_type,
                        input_dim=operator_feature_dim,
                        num_children=num_children,
                        data_vec_dim=data_vec_dim,
                        hidden_dim=self.hidden_dim,
                        num_layers=self.num_layers
                    )
                    # Add to model's units
                    key = self.model.get_unit_key(operator_type, num_children)
                    self.model.units[key] = unit
                    self.model.operator_types.add(operator_type)

        print('Loading trained weights...')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        print(f'  Trained for {checkpoint["epoch"]} epochs')
        print(f'  Model has {self.model.count_parameters():,} parameters')
        print(f'  Number of neural units: {len(self.model.units)}')

    def close(self):
        """Close database connection."""
        self.extractor.close()

    def predict_latency(self, query: str) -> tuple[float, dict]:
        """
        Predict query latency using EXPLAIN

        Args:
            query: Cypher query string

        Returns:
            Tuple of (predicted_latency, plan)
        """
        # Get plan with EXPLAIN
        plan = self.extractor.get_plan(query)

        with torch.no_grad():
            predicted_latency = self.model.forward(plan)

        return predicted_latency.item(), plan

    def measure_actual_latency(self, query: str, num_runs: int = 3) -> tuple[float, float]:
        """
        Measure actual query execution time.

        Args:
            query: Cypher query string
            num_runs: Number of executions for averaging

        Returns:
            Tuple of (mean_latency, std_latency)
        """
        execution_times = []

        with self.extractor.neo4j.session() as session:
            for _ in range(num_runs):
                start_time = time.time()
                result = session.run(cypher(query))
                result.consume()
                end_time = time.time()
                execution_times.append(end_time - start_time)

        return np.mean(execution_times).item(), np.std(execution_times).item()

    def evaluate_query(self, query: TestQuery, num_runs: int = 3) -> 'Result':
        """
        Evaluate a single query.

        Args:
            num_runs: Number of executions for averaging
        """
        print(f'\nEvaluating: {query.name}')

        result = Result(query.name, query.content)
        predicted_latency, result.plan = self.predict_latency(query.content)
        actual_latency, result.std_latency = self.measure_actual_latency(query.content, num_runs)

        result.predicted_latency = predicted_latency
        result.actual_latency = actual_latency

        # Compute metrics
        result.absolute_error = abs(predicted_latency - actual_latency)
        result.r_value = max(predicted_latency / actual_latency, actual_latency / predicted_latency) \
                  if predicted_latency > 0 and actual_latency > 0 else float('inf')

        print(f'  Predicted: {predicted_latency * 1000:.2f}ms')
        print(f'  Actual: {actual_latency * 1000:.2f}ms (±{result.std_latency * 1000:.2f}ms)')
        print(f'  Absolute Error: {result.absolute_error * 1000:.2f}ms')
        print(f'  R-value: {result.r_value:.4f}')

        return result

    def evaluate_multiple_queries(self, queries: list[TestQuery], num_runs: int = 3) -> list['Result']:
        """
        Evaluate multiple queries.

        Args:
            num_runs: Number of executions per query for averaging
        """
        results: list['Result'] = []

        print(f'\nEvaluating {len(queries)} queries...')
        print('=' * 70)

        for query in queries:
            try:
                result = self.evaluate_query(query, num_runs)
                results.append(result)
            except Exception as e:
                print(f'  ✗ Error evaluating {query.name}: {str(e)}')
                continue

        return results

    def print_summary(self, results: list['Result']):
        """Print summary statistics of evaluation results."""
        if not results:
            print('\nNo results to summarize.')
            return

        print('\n' + '=' * 70)
        print('Evaluation Summary')
        print('=' * 70)

        # Extract metrics
        absolute_errors = [r.absolute_error for r in results]
        r_values = [r.r_value for r in results if r.r_value != float('inf')]

        # Compute statistics
        print(f'\nNumber of queries: {len(results)}')
        print(f'\nAbsolute Error:')
        print(f'  Mean: {np.mean(absolute_errors) * 1000:.2f}ms')
        print(f'  Median: {np.median(absolute_errors) * 1000:.2f}ms')
        print(f'  Std: {np.std(absolute_errors) * 1000:.2f}ms')
        print(f'  Min/Max: {np.min(absolute_errors) * 1000:.2f}ms / {np.max(absolute_errors) * 1000:.2f}ms')


        if r_values:
            print(f'\nR-value:')
            print(f'  Mean: {np.mean(r_values):.4f}')
            print(f'  Median: {np.median(r_values):.4f}')
            print(f'  90th percentile: {np.percentile(r_values, 90):.4f}')
            print(f'  95th percentile: {np.percentile(r_values, 95):.4f}')
            print(f'  Min/Max: {np.min(r_values):.4f} / {np.max(r_values):.4f}')

        # Create results table
        table_data = []
        for r in results:
            table_data.append([
                r.name[:30] if r.name else 'N/A',
                f'{r.predicted_latency * 1000:.2f}',
                f'{r.actual_latency * 1000:.2f}',
                f'{r.absolute_error * 1000:.2f}',
                f'{r.r_value:.4f}' if r.r_value != float('inf') else 'inf'
            ])

        print('\n' + '=' * 70)
        print('Detailed Results')
        print('=' * 70)
        print(tabulate(
            table_data,
            headers=['Query', 'Predicted (ms)', 'Actual (ms)', 'Abs Error (ms)', 'R-value'],
            tablefmt='grid'
        ))

        r_value_thresholds = [1.5, 2.0, 3.0]
        total_queries = len(results)
        print('\nR-value Thresholds:')
        for threshold in r_value_thresholds:
            count_below = sum(1 for r in results if r.r_value < threshold)
            percent_below = (count_below / total_queries) * 100
            print(f'  R-value < {threshold}: {count_below} queries ({percent_below:.2f}%)')

class Result:
    """Holds a single query evaluation result."""
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

        self.predicted_latency: float
        self.actual_latency: float
        self.std_latency: float
        self.absolute_error: float
        self.r_value: float
        self.plan: dict

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Neo4j QPP model')
    parser.add_argument('--checkpoint', type=str, default='data/neo4j_qpp_checkpoint_best.pt', help='Path to model checkpoint')
    parser.add_argument('--feature-extractor', type=str, default='data/neo4j_qpp_checkpoint_feature_extractor.pkl', help='Path to feature extractor')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of executions per query for averaging')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference')
    parser.add_argument('--num-layers', type=int, default=10, help='Number of layers in the neural units')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size in the neural units')
    parser.add_argument('--query', '-q', type=str, action='append', dest='queries', help='Additional Cypher query to evaluate (can be used multiple times)')
    parser.add_argument('--query-only', '-qo', action='store_true', help='Only evaluate the provided --query arguments, skip built-in test queries')

    args = parser.parse_args()

    print('=' * 70)
    print('Neo4j Query Performance Predictor - Evaluation')
    print('=' * 70)

    # Load model
    config = Config.load()
    neo4j = Neo4jDriver(config.neo4j)
    database = TpchNeo4j()
    evaluator = ModelEvaluator(
        neo4j=neo4j,
        database=database,
        checkpoint_path=args.checkpoint,
        feature_extractor_path=args.feature_extractor,
        device=args.device,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim
    )

    try:
        # Generate test queries based on arguments
        print('\nGenerating test queries...')
        test_queries: list[TestQuery] = [] if args.query_only else database.get_test_queries()

        # Add user-provided queries
        if args.queries:
            for i, query in enumerate(args.queries, 1):
                query = TestQuery(f'Custom Query {i}', query)
                test_queries.append(query)
            print(f'Added {len(args.queries)} custom query/queries')

        if not test_queries:
            print('Error: No queries to evaluate. Provide queries with --query or remove --query-only flag.')
            return

        print(f'Total queries to evaluate: {len(test_queries)}')

        results = evaluator.evaluate_multiple_queries(test_queries, args.num_runs)
        evaluator.print_summary(results)


    finally:
        evaluator.close()

if __name__ == '__main__':
    main()
