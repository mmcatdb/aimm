from math import nan
import torch
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import json
from common.utils import EPSILON
from common.database import TestQuery
from latency_estimation.postgres.plan_extractor import PlanExtractor
from latency_estimation.postgres.plan_structured_network import PlanStructuredNetwork

class ModelEvaluator:
    """Evaluates a trained QPP-Net model on new queries."""
    def __init__(self, extractor: PlanExtractor, model: PlanStructuredNetwork):
        self.extractor = extractor
        self.model = model

    def evaluate_multiple_queries(self, queries: list[TestQuery], measure_actual: bool, num_runs: int) -> list['Result']:
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
                print(f'\nError evaluating {query.label()}: {e}')
                import traceback
                traceback.print_exc()
                continue

        return results

    def evaluate_query(self, query: TestQuery, measure_actual: bool, num_runs: int) -> 'Result':
        """
        Comprehensive evaluation of a single query.
        Args:
            measure_actual: Whether to measure actual execution time
            num_runs: Number of runs for actual execution measurement
        Returns:
            Result object with all measurements and comparisons
        """
        print(f'\nEvaluating: {query.label()}')
        print('-' * 80)

        result = Result(query.label(), query.content)

        # 1. Get EXPLAIN ANALYZE time
        print('  [1/3] Running EXPLAIN ANALYZE...')
        plan, explain_time = self.extractor.explain_plan_and_measure(query.content, False)
        result.explain_analyze_time = explain_time
        print(f'        EXPLAIN ANALYZE: {explain_time:.2f} ms')

        # 2. Get estimation (without executing)
        print('  [2/3] Getting query plan and estimation...')
        estimated_time = self.__estimate_latency(plan)
        result.estimated_time = estimated_time
        print(f'        Model Estimation: {estimated_time:.2f} ms')

        # Calculate errors and ratios
        result.error_vs_explain = abs(estimated_time - explain_time)
        result.relative_error_vs_explain = abs(estimated_time - explain_time) / (explain_time + EPSILON)
        result.r_vs_explain = max(estimated_time / (explain_time + EPSILON), explain_time / (estimated_time + EPSILON))
        result.explain_estimated_ratio = explain_time / (estimated_time + EPSILON)
        print(f'\n  Estimation Error vs EXPLAIN ANALYZE: {result.error_vs_explain:.2f} ms (R={result.r_vs_explain:.2f})')

        # 3. Get actual execution time
        if measure_actual:
            actual = ActualResult()
            result.actual = actual

            print(f'  [3/3] Measuring actual execution ({num_runs} runs)...')
            actual_mean, actual_min, actual_max, _ = self.extractor.measure_query(query.content, num_runs)
            actual.time_mean = actual_mean
            actual.time_min = actual_min
            actual.time_max = actual_max
            print(f'        Actual Time: {actual_mean:.2f} ms (min: {actual_min:.2f}, max: {actual_max:.2f})')

            actual.error_vs_actual = abs(estimated_time - actual_mean)
            actual.relative_error_vs_actual = abs(estimated_time - actual_mean) / (actual_mean + EPSILON)
            actual.r_vs_actual = max(estimated_time / (actual_mean + EPSILON), actual_mean / (estimated_time + EPSILON))
            actual.estimated_ratio = actual_mean / (estimated_time + EPSILON)

            # Also compare EXPLAIN ANALYZE vs Actual
            actual.explain_vs_actual_error = abs(explain_time - actual_mean)
            actual.explain_vs_actual_relative = abs(explain_time - actual_mean) / (actual_mean + EPSILON)

            print(f'  Estimation Error vs Actual: {actual.error_vs_actual:.2f} ms (R={actual.r_vs_actual:.2f})')
            print(f'  EXPLAIN ANALYZE vs Actual: {actual.explain_vs_actual_error:.2f} ms')

        return result

    def __estimate_latency(self, plan: dict) -> float:
        with torch.no_grad():
            return self.model(plan).item()

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
                f'{r.estimated_time:.1f}',
                f'{r.explain_analyze_time:.1f}',
                f'{r.actual.time_mean:.1f}' if r.actual else 'N/A',
                f'{r.explain_estimated_ratio:.2f}',
                f'{r.r_vs_explain:.2f}',
            ]
            if r.actual:
                row.extend([
                    f'{r.actual.estimated_ratio:.2f}',
                    f'{r.actual.r_vs_actual:.2f}',
                ])
            table_data.append(row)

        headers = [
            'Query',
            'Estimated\n(ms)',
            'EXPLAIN\n(ms)',
            'Actual\n(ms)',
            'Explain/\nEstimated',
            'R vs\nEXPLAIN'
        ]

        if results[0].actual:
            headers.extend(['Actual/\nEstimated', 'R vs\nActual'])

        print('\n' + tabulate(table_data, headers=headers, tablefmt='grid'))

        # Summary statistics
        print('\n' + '=' * 80)
        print('AGGREGATE STATISTICS')
        print('=' * 80)

        # Estimation vs EXPLAIN ANALYZE
        errors_explain = [r.error_vs_explain for r in results]
        relative_errors_explain = [r.relative_error_vs_explain for r in results]
        r_values_explain = [r.r_vs_explain for r in results]

        print('\nModel Estimation vs EXPLAIN ANALYZE:')
        print(f'  Mean Absolute Error: {np.mean(errors_explain):.2f} ms')
        print(f'  Median Absolute Error: {np.median(errors_explain):.2f} ms')
        print(f'  Mean Relative Error: {np.mean(relative_errors_explain):.4f}')
        print(f'  Median R-value: {np.median(r_values_explain):.2f}')
        print(f'  R ≤ 1.5: {np.mean([r <= 1.5 for r in r_values_explain]) * 100:.1f}%')
        print(f'  R ≤ 2.0: {np.mean([r <= 2.0 for r in r_values_explain]) * 100:.1f}%')

        # Estimation vs Actual
        if results[0].actual:
            ar = [r.actual for r in results if r.actual is not None]
            errors_actual = [r.error_vs_actual for r in ar]
            relative_errors_actual = [r.relative_error_vs_actual for r in ar]
            r_values_actual = [r.r_vs_actual for r in ar]

            print('\nModel Estimation vs Actual Execution:')
            print(f'  Mean Absolute Error: {np.mean(errors_actual):.2f} ms')
            print(f'  Median Absolute Error: {np.median(errors_actual):.2f} ms')
            print(f'  Mean Relative Error: {np.mean(relative_errors_actual):.4f}')
            print(f'  Median R-value: {np.median(r_values_actual):.2f}')
            print(f'  R ≤ 1.5: {np.mean([r <= 1.5 for r in r_values_actual]) * 100:.1f}%')
            print(f'  R ≤ 2.0: {np.mean([r <= 2.0 for r in r_values_actual]) * 100:.1f}%')

            # EXPLAIN ANALYZE vs Actual
            explain_vs_actual = [r.explain_vs_actual_error for r in ar]
            explain_vs_actual_rel = [r.explain_vs_actual_relative for r in ar]

            print('\nEXPLAIN ANALYZE vs Actual Execution:')
            print(f'  Mean Absolute Error: {np.mean(explain_vs_actual):.2f} ms')
            print(f'  Median Absolute Error: {np.median(explain_vs_actual):.2f} ms')
            print(f'  Mean Relative Error: {np.mean(explain_vs_actual_rel):.4f}')

    def plot_results(self, results: list['Result'], save_path: str):
        """Create visualization plots comparing estimations and actual times."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('QPP-Net Model Evaluation', fontsize=16, fontweight='bold')

        estimated = [r.estimated_time for r in results]
        explain = [r.explain_analyze_time for r in results]

        # Plot 1: Estimated vs EXPLAIN ANALYZE
        ax1 = axes[0, 0]
        ax1.scatter(explain, estimated, alpha=0.6)
        min_val = min(min(explain), min(estimated))
        max_val = max(max(explain), max(estimated))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Estimation')
        ax1.set_xlabel('EXPLAIN ANALYZE Time (ms)')
        ax1.set_ylabel('Estimated Time (ms)')
        ax1.set_title('Estimation vs EXPLAIN ANALYZE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error distribution (Estimated vs EXPLAIN)
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

            # Plot 3: Estimated vs Actual
            ax3 = axes[1, 0]
            ax3.scatter(actual, estimated, alpha=0.6, color='green')
            min_val = min(min(actual), min(estimated))
            max_val = max(max(actual), max(estimated))
            ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Estimation')
            ax3.set_xlabel('Actual Execution Time (ms)')
            ax3.set_ylabel('Estimated Time (ms)')
            ax3.set_title('Estimation vs Actual Execution')
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

@dataclass
class ActualResult:
    time_mean: float = nan
    time_min: float = nan
    time_max: float = nan
    error_vs_actual: float = nan
    relative_error_vs_actual: float = nan
    r_vs_actual: float = nan
    estimated_ratio: float = nan
    explain_vs_actual_error: float = nan
    explain_vs_actual_relative: float = nan

@dataclass
class Result:
    """Holds a single query evaluation result."""
    name: str
    content: str
    estimated_time: float = nan
    explain_analyze_time: float = nan
    error_vs_explain: float = nan
    relative_error_vs_explain: float = nan
    r_vs_explain: float = nan
    explain_estimated_ratio: float = nan
    actual: ActualResult | None = None
