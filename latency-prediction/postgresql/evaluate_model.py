from re import T
from typing import cast
import torch
import numpy as np
import time
import json
import pickle
from tabulate import tabulate
import matplotlib.pyplot as plt
from datasets.database import TestQuery
from datasets.tpch.postgres import TpchPostgres
from common.config import Config
from common.drivers import PostgresDriver
from plan_structured_network import PlanStructuredNetwork

class ModelEvaluator:
    """Evaluates a trained QPP-Net model on new queries."""

    def __init__(self, postgres: PostgresDriver, model_path: str = 'qpp_net_model.pt'):
        """
        Load trained model and initialize evaluator.

        Args:
            model_path: Path to saved model checkpoint
        """
        self.postgres = postgres

        print('Loading trained model...')
        checkpoint = torch.load(model_path, weights_only=False)

        self.feature_extractor = checkpoint['feature_extractor']
        config = checkpoint['config']

        # Recreate model architecture
        self.model = PlanStructuredNetwork(
            feature_extractor=self.feature_extractor,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            data_vec_dim=config['data_vec_dim']
        )

        # Initialize units from saved operator info
        if 'operator_info' in checkpoint:
            self.model.initialize_units_from_operator_info(checkpoint['operator_info'])
        else:
            raise ValueError('Model checkpoint does not contain operator_info. Please retrain the model with the updated main.py script.')

        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Print training metrics if available
        if 'metrics' in checkpoint:
            print('\nTraining Metrics:')
            print(f'  Train MAE: {checkpoint["metrics"]["train"]["mae"]:.2f} ms')
            print(f'  Test MAE: {checkpoint["metrics"]["test"]["mae"]:.2f} ms')
            print(f'  Test R ≤ 1.5: {checkpoint["metrics"]["test"]["r_within_1.5"]*100:.1f}%')

        print('Model loaded successfully!\n')

    def get_query_plan_only(self, query: str) -> dict:
        """
        Get query plan WITHOUT executing the query.
        Uses EXPLAIN (without ANALYZE).

        Args:
            query: SQL query string

        Returns:
            Query plan dictionary
        """
        connection = self.postgres.get_connection()
        connection.autocommit = True

        try:
            with connection.cursor() as cursor:
                explain_query = f'EXPLAIN (FORMAT JSON, VERBOSE) {query}'
                cursor.execute(explain_query)
                result = cursor.fetchone()
                assert result is not None, 'No plan returned from EXPLAIN.'

                plan = result[0][0]['Plan']
                return plan
        finally:
            self.postgres.put_connection(connection)

    def get_explain_analyze_time(self, query: str) -> tuple[dict, float]:
        """
        Get query plan and execution time from EXPLAIN ANALYZE.

        Args:
            query: SQL query string

        Returns:
            Tuple of (plan, explain_analyze_time_ms)
        """
        connection = self.postgres.get_connection()
        connection.autocommit = True

        try:
            with connection.cursor() as cursor:
                explain_query = f'EXPLAIN (ANALYZE, FORMAT JSON, BUFFERS, VERBOSE) {query}'
                cursor.execute(explain_query)
                result = cursor.fetchone()
                assert result is not None, 'No plan returned from EXPLAIN ANALYZE.'

                plan_json = result[0][0]
                execution_time = plan_json['Execution Time']
                return plan_json['Plan'], execution_time
        finally:
            self.postgres.put_connection(connection)

    def get_actual_execution_time(self, query: str, num_runs: int = 3) -> tuple[float, float, float]:
        """
        Execute query and measure actual wall-clock time.
        Runs multiple times and returns statistics.

        Args:
            query: SQL query string
            num_runs: Number of times to execute the query

        Returns:
            Tuple of (mean_time_ms, min_time_ms, max_time_ms)
        """
        times = []

        for i in range(num_runs):
            connection = self.postgres.get_connection()
            connection.autocommit = True

            try:
                with connection.cursor() as cursor:
                    start_time = time.time()
                    cursor.execute(query)
                    # Fetch all results to ensure query completes
                    cursor.fetchall()
                    end_time = time.time()

                    elapsed_ms = (end_time - start_time) * 1000
                    times.append(elapsed_ms)
            finally:
                self.postgres.put_connection(connection)

        return np.mean(times).item(), np.min(times), np.max(times)

    def predict_latency(self, plan: dict) -> float:
        """
        Predict query latency using the trained model.

        Args:
            plan: Query plan dictionary

        Returns:
            Predicted latency in milliseconds
        """
        with torch.no_grad():
            prediction = self.model(plan).item()
        return prediction

    def evaluate_query(self, query: TestQuery, measure_actual: bool = True, num_runs: int = 3) -> 'Result':
        """
        Comprehensive evaluation of a single query.

        Args:
            measure_actual: Whether to measure actual execution time
            num_runs: Number of runs for actual execution measurement

        Returns:
            Result object with all measurements and comparisons
        """
        print(f'\nEvaluating: {query.name}')
        print('-' * 80)

        result = Result(query.name, query.content)

        # 1. Get prediction (without executing)
        print('  [1/3] Getting query plan and prediction...')
        plan = self.get_query_plan_only(query.content)
        predicted_time = self.predict_latency(plan)
        result.predicted_time = predicted_time
        print(f'        Model Prediction: {predicted_time:.2f} ms')

        # 2. Get EXPLAIN ANALYZE time
        print('  [2/3] Running EXPLAIN ANALYZE...')
        _, explain_time = self.get_explain_analyze_time(query.content)
        result.explain_analyze_time = explain_time
        print(f'        EXPLAIN ANALYZE: {explain_time:.2f} ms')

        # Calculate errors and ratios
        result.error_vs_explain = abs(predicted_time - explain_time)
        result.relative_error_vs_explain = abs(predicted_time - explain_time) / (explain_time + 1e-8)
        result.r_vs_explain = max(predicted_time / (explain_time + 1e-8), explain_time / (predicted_time + 1e-8))
        result.explain_predicted_ratio = explain_time / (predicted_time + 1e-8)
        print(f'\n  Prediction Error vs EXPLAIN ANALYZE: {result.error_vs_explain:.2f} ms (R={result.r_vs_explain:.2f})')

        # 3. Get actual execution time
        if measure_actual:
            actual = ActualResult()
            result.actual = actual

            print(f'  [3/3] Measuring actual execution ({num_runs} runs)...')
            actual_mean, actual_min, actual_max = self.get_actual_execution_time(query.content, num_runs)
            actual.time_mean = actual_mean
            actual.time_min = actual_min
            actual.time_max = actual_max
            print(f'        Actual Time: {actual_mean:.2f} ms (min: {actual_min:.2f}, max: {actual_max:.2f})')

            actual.error_vs_actual = abs(predicted_time - actual_mean)
            actual.relative_error_vs_actual = abs(predicted_time - actual_mean) / (actual_mean + 1e-8)
            actual.r_vs_actual = max(predicted_time / (actual_mean + 1e-8), actual_mean / (predicted_time + 1e-8))
            actual.predicted_ratio = actual_mean / (predicted_time + 1e-8)

            # Also compare EXPLAIN ANALYZE vs Actual
            actual.explain_vs_actual_error = abs(explain_time - actual_mean)
            actual.explain_vs_actual_relative = abs(explain_time - actual_mean) / (actual_mean + 1e-8)

            print(f'  Prediction Error vs Actual: {actual.error_vs_actual:.2f} ms (R={actual.r_vs_actual:.2f})')
            print(f'  EXPLAIN ANALYZE vs Actual: {actual.explain_vs_actual_error:.2f} ms')

        return result

    def evaluate_multiple_queries(self, queries: list[TestQuery], measure_actual: bool = True, num_runs: int = 3) -> list['Result']:
        """
        Evaluate multiple queries.

        Args:
            measure_actual: Whether to measure actual execution times
            num_runs: Number of runs for each query
        """
        results: list['Result'] = []

        print('=' * 80)
        print(f'EVALUATING {len(queries)} QUERIES')
        print('=' * 80)

        for query in queries:
            try:
                result = self.evaluate_query(query, measure_actual, num_runs)
                results.append(result)
            except Exception as e:
                print(f'\nError evaluating {query.name}: {e}')
                import traceback
                traceback.print_exc()
                continue

        return results

    def print_summary(self, results: list['Result']):
        """Print summary statistics and comparison table."""

        print('\n' + '=' * 80)
        print('EVALUATION SUMMARY')
        print('=' * 80)

        # Prepare data for table
        table_data = []
        for r in results:
            row = [
                r.name,
                f'{r.predicted_time:.1f}',
                f'{r.explain_analyze_time:.1f}',
                f'{r.actual.time_mean:.1f}' if r.actual else 'N/A',
                f'{r.explain_predicted_ratio:.2f}',
                f'{r.r_vs_explain:.2f}',
            ]
            if r.actual:
                row.extend([
                    f'{r.actual.predicted_ratio:.2f}',
                    f'{r.actual.r_vs_actual:.2f}',
                ])
            table_data.append(row)

        headers = [
            'Query',
            'Predicted\n(ms)',
            'EXPLAIN\n(ms)',
            'Actual\n(ms)',
            'Explain/\nPredicted',
            'R vs\nEXPLAIN'
        ]

        if results[0].actual:
            headers.extend(['Actual/\nPredicted', 'R vs\nActual'])

        print('\n' + tabulate(table_data, headers=headers, tablefmt='grid'))

        # Summary statistics
        print('\n' + '=' * 80)
        print('AGGREGATE STATISTICS')
        print('=' * 80)

        # Prediction vs EXPLAIN ANALYZE
        errors_explain = [r.error_vs_explain for r in results]
        relative_errors_explain = [r.relative_error_vs_explain for r in results]
        r_values_explain = [r.r_vs_explain for r in results]

        print('\nModel Prediction vs EXPLAIN ANALYZE:')
        print(f'  Mean Absolute Error: {np.mean(errors_explain):.2f} ms')
        print(f'  Median Absolute Error: {np.median(errors_explain):.2f} ms')
        print(f'  Mean Relative Error: {np.mean(relative_errors_explain):.4f}')
        print(f'  Median R-value: {np.median(r_values_explain):.2f}')
        print(f'  R ≤ 1.5: {np.mean([r <= 1.5 for r in r_values_explain])*100:.1f}%')
        print(f'  R ≤ 2.0: {np.mean([r <= 2.0 for r in r_values_explain])*100:.1f}%')

        # Prediction vs Actual
        if results[0].actual:
            ar = [r.actual for r in results if r.actual is not None]
            errors_actual = [r.error_vs_actual for r in ar]
            relative_errors_actual = [r.relative_error_vs_actual for r in ar]
            r_values_actual = [r.r_vs_actual for r in ar]

            print('\nModel Prediction vs Actual Execution:')
            print(f'  Mean Absolute Error: {np.mean(errors_actual):.2f} ms')
            print(f'  Median Absolute Error: {np.median(errors_actual):.2f} ms')
            print(f'  Mean Relative Error: {np.mean(relative_errors_actual):.4f}')
            print(f'  Median R-value: {np.median(r_values_actual):.2f}')
            print(f'  R ≤ 1.5: {np.mean([r <= 1.5 for r in r_values_actual])*100:.1f}%')
            print(f'  R ≤ 2.0: {np.mean([r <= 2.0 for r in r_values_actual])*100:.1f}%')

            # EXPLAIN ANALYZE vs Actual
            explain_vs_actual = [r.explain_vs_actual_error for r in ar]
            explain_vs_actual_rel = [r.explain_vs_actual_relative for r in ar]

            print('\nEXPLAIN ANALYZE vs Actual Execution:')
            print(f'  Mean Absolute Error: {np.mean(explain_vs_actual):.2f} ms')
            print(f'  Median Absolute Error: {np.median(explain_vs_actual):.2f} ms')
            print(f'  Mean Relative Error: {np.mean(explain_vs_actual_rel):.4f}')

    def plot_results(self, results: list['Result'], save_path: str = 'evaluation_plots.png'):
        """Create visualization plots comparing predictions and actual times."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('QPP-Net Model Evaluation', fontsize=16, fontweight='bold')

        predicted = [r.predicted_time for r in results]
        explain = [r.explain_analyze_time for r in results]

        # Plot 1: Predicted vs EXPLAIN ANALYZE
        ax1 = axes[0, 0]
        ax1.scatter(explain, predicted, alpha=0.6)
        min_val = min(min(explain), min(predicted))
        max_val = max(max(explain), max(predicted))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax1.set_xlabel('EXPLAIN ANALYZE Time (ms)')
        ax1.set_ylabel('Predicted Time (ms)')
        ax1.set_title('Prediction vs EXPLAIN ANALYZE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error distribution (Predicted vs EXPLAIN)
        ax2 = axes[0, 1]
        errors = [r.error_vs_explain for r in results]
        ax2.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Absolute Error (ms)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution (vs EXPLAIN ANALYZE)')
        ax2.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.1f}ms')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        if results[0].actual:
            actual = [r.actual.time_mean for r in results if r.actual is not None]

            # Plot 3: Predicted vs Actual
            ax3 = axes[1, 0]
            ax3.scatter(actual, predicted, alpha=0.6, color='green')
            min_val = min(min(actual), min(predicted))
            max_val = max(max(actual), max(predicted))
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            ax3.set_xlabel('Actual Execution Time (ms)')
            ax3.set_ylabel('Predicted Time (ms)')
            ax3.set_title('Prediction vs Actual Execution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: EXPLAIN ANALYZE vs Actual
            ax4 = axes[1, 1]
            ax4.scatter(actual, explain, alpha=0.6, color='orange')
            min_val = min(min(actual), min(explain))
            max_val = max(max(actual), max(explain))
            ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Match')
            ax4.set_xlabel('Actual Execution Time (ms)')
            ax4.set_ylabel('EXPLAIN ANALYZE Time (ms)')
            ax4.set_title('EXPLAIN ANALYZE vs Actual Execution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Plot 3: R-value distribution
            ax3 = axes[1, 0]
            r_values = [r.r_vs_explain for r in results]
            ax3.hist(r_values, bins=20, edgecolor='black', alpha=0.7)
            ax3.set_xlabel('R-value')
            ax3.set_ylabel('Frequency')
            ax3.set_title('R-value Distribution (vs EXPLAIN ANALYZE)')
            ax3.axvline(1.5, color='r', linestyle='--', label='R=1.5 threshold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

            # Hide plot 4
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'\nPlots saved to: {save_path}')

        return fig

class ActualResult:
    def __init__(self):
        self.time_mean: float
        self.time_min: float
        self.time_max: float
        self.error_vs_actual: float
        self.relative_error_vs_actual: float
        self.r_vs_actual: float
        self.predicted_ratio: float
        self.explain_vs_actual_error: float
        self.explain_vs_actual_relative: float

class Result:
    """Holds a single query evaluation result."""
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content

        self.predicted_time: float
        self.explain_analyze_time: float
        self.error_vs_explain: float
        self.relative_error_vs_explain: float
        self.r_vs_explain: float
        self.explain_predicted_ratio: float

        self.actual: ActualResult | None = None

def main():
    """Main evaluation routine."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained QPP-Net model')
    parser.add_argument('--model', type=str, default='qpp_net_model.pt', help='Path to trained model')
    parser.add_argument('--no-actual', action='store_true', help='Skip actual execution time measurement')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs for actual execution measurement')
    parser.add_argument('--save-results', type=str, default='evaluation_results.json', help='Path to save results JSON')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--config', type=str, default=None, help=f'Path to config file (default: {Config.DEFAULT_CONFIG_PATH})')
    parser.add_argument('--query', '-q', type=str, action='append', dest='queries', help='Additional SQL query to evaluate (can be used multiple times)')
    parser.add_argument('--query-only', '-qo', action='store_true', help='Only evaluate the provided --query arguments, skip built-in test queries')

    args = parser.parse_args()

    print('=' * 80)
    print('QPP-NET MODEL EVALUATION')
    print('=' * 80)

    # Initialize evaluator
    config = Config.load(args.config)
    postgres = PostgresDriver(config.postgres)
    evaluator = ModelEvaluator(postgres, args.model)
    database = TpchPostgres()

    # Generate test queries
    print('\nGenerating test queries...')
    test_queries: list[TestQuery] = [] if args.query_only else database.get_test_queries()

    # Add user-provided queries
    if args.queries:
        for i, content in enumerate(args.queries, 1):
            query = TestQuery(f'Custom Query {i}', content)
            test_queries.append(query)
        print(f'Added {len(args.queries)} custom query/queries')

    if not test_queries:
        print('Error: No queries to evaluate. Provide queries with --query or remove --query-only flag.')
        return

    print(f'Total queries to evaluate: {len(test_queries)}')

    # Run evaluation
    results = evaluator.evaluate_multiple_queries(
        test_queries,
        measure_actual=not args.no_actual,
        num_runs=args.runs
    )

    # Print summary
    evaluator.print_summary(results)

    # Save results
    print(f'\nSaving results to {args.save_results}...')
    with open(args.save_results, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate plots
    if not args.no_plots:
        try:
            evaluator.plot_results(results)
        except Exception as e:
            print(f'Warning: Could not generate plots: {e}')

    print('\n' + '=' * 80)
    print('EVALUATION COMPLETE!')
    print('=' * 80)

if __name__ == '__main__':
    main()
