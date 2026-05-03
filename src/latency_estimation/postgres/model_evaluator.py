from typing_extensions import override
import numpy as np
from tabulate import tabulate
from ..model_evaluator import BaseModelEvaluator, ExtractedQueryResult, QueryResult
from .plan_extractor import PlanExtractor

class ModelEvaluator(BaseModelEvaluator):

    @override
    def _extract_latency_from_plan(self, plan: dict) -> float | None:
        return plan[PlanExtractor.EXECUTION_TIME_KEY]

    @override
    def print_summary(self, results: list[QueryResult]):
        extracted_results = self._get_extracted_results(results)
        if not extracted_results:
            print('\nNo successful query results to summarize.')
            return

        self.__inner_print_summary(extracted_results)

    def __inner_print_summary(self, results: list[ExtractedQueryResult]):
        print('\n' + '=' * 80)
        print('EVALUATION SUMMARY')
        print('=' * 80)

        # Prepare data for table
        table_data = []
        for r in results:
            row = [
                r.label[:30],
                f'{r.predicted:.1f}',
                f'{r.extracted:.1f}',
                f'{r.measured.mean:.1f}',
                f'{r.vs_extracted.ratio:.2f}',
                f'{r.vs_extracted.r_value:.2f}',
                f'{r.vs_measured.ratio:.2f}',
                f'{r.vs_measured.r_value:.2f}',
            ]
            table_data.append(row)

        headers = [
            'Query',
            'Estimated\n(ms)',
            'EXPLAIN\n(ms)',
            'Actual\n(ms)',
            'Explain/\nEstimated',
            'R vs\nEXPLAIN',
            'Actual/\nEstimated',
            'R vs\nActual',
        ]

        print('\n' + '=' * 80)
        print('Detailed Results')
        print('=' * 80)

        print('\n' + tabulate(table_data, headers=headers, tablefmt='grid'))

        # Summary statistics
        print('\n' + '=' * 80)
        print('AGGREGATE STATISTICS')
        print('=' * 80)

        print('\nModel Estimation vs EXPLAIN ANALYZE:')
        vs_extracted = [r.vs_extracted for r in results]
        self._summary_diff_results(vs_extracted, r_value_thresholds=[1.5, 2.0])

        vs_measured = [r.vs_measured for r in results]
        print('\nModel Estimation vs Actual Execution:')
        self._summary_diff_results(vs_measured, r_value_thresholds=[1.5, 2.0])

        extracted_vs_measured = [r.extracted_vs_measured for r in results]
        print('\nEXPLAIN ANALYZE vs Actual Execution:')
        self._summary_diff_results(extracted_vs_measured)

    def plot_results(self, results: list[QueryResult], save_path: str):
        extracted_results = self._get_extracted_results(results)
        if not extracted_results:
            print('\nNo successful query results to plot.')
            return None

        return self.__plot_results_inner(extracted_results, save_path)

    def __plot_results_inner(self, results: list[ExtractedQueryResult], save_path: str):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('QPP-Net Model Evaluation', fontsize=16, fontweight='bold')

        predicted = [r.predicted for r in results]
        extracted = [r.extracted for r in results]

        # Plot 1: Estimated vs EXPLAIN ANALYZE
        ax1 = axes[0, 0]
        ax1.scatter(extracted, predicted, alpha=0.6)
        min_val = min(min(extracted), min(predicted))
        max_val = max(max(extracted), max(predicted))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Estimation')
        ax1.set_xlabel('EXPLAIN ANALYZE Time (ms)')
        ax1.set_ylabel('Estimated Time (ms)')
        ax1.set_title('Estimation vs EXPLAIN ANALYZE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error distribution (Estimated vs EXPLAIN)
        ax2 = axes[0, 1]
        errors = [r.vs_extracted.absolute for r in results]
        ax2.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Absolute Error (ms)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution (vs EXPLAIN ANALYZE)')
        ax2.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.1f}ms')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        actual = [r.measured.mean for r in results]

        # Plot 3: Estimated vs Actual
        ax3 = axes[1, 0]
        ax3.scatter(actual, predicted, alpha=0.6, color='green')
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Estimation')
        ax3.set_xlabel('Actual Execution Time (ms)')
        ax3.set_ylabel('Estimated Time (ms)')
        ax3.set_title('Estimation vs Actual Execution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: EXPLAIN ANALYZE vs Actual
        ax4 = axes[1, 1]
        ax4.scatter(actual, extracted, alpha=0.6, color='orange')
        min_val = min(min(actual), min(extracted))
        max_val = max(max(actual), max(extracted))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Match')
        ax4.set_xlabel('Actual Execution Time (ms)')
        ax4.set_ylabel('EXPLAIN ANALYZE Time (ms)')
        ax4.set_title('EXPLAIN ANALYZE vs Actual Execution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

            # If not actual?
            # # Plot 3: R-value distribution
            # ax3 = axes[1, 0]
            # r_values = [r.vs_extracted.r_value for r in results]
            # ax3.hist(r_values, bins=20, edgecolor='black', alpha=0.7)
            # ax3.set_xlabel('R-value')
            # ax3.set_ylabel('Frequency')
            # ax3.set_title('R-value Distribution (vs EXPLAIN ANALYZE)')
            # ax3.axvline(1.5, color='r', linestyle='--', label='R=1.5 threshold')
            # ax3.legend()
            # ax3.grid(True, alpha=0.3, axis='y')

            # # Hide plot 4
            # axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'\nPlots saved to: {save_path}')

        return fig
