from typing_extensions import override
import numpy as np
from tabulate import tabulate
from ..model_evaluator import BaseModelEvaluator, QueryResult

class ModelEvaluator(BaseModelEvaluator):

    @override
    def print_summary(self, results: list[QueryResult]):
        print('\n' + '=' * 80)
        print('Evaluation Summary')
        print('=' * 80)

        # Extract metrics
        absolute = [r.vs_measured.absolute for r in results]
        r_value = [r.vs_measured.r_value for r in results]

        # Compute statistics
        print(f'\nNumber of queries: {len(results)}')
        print(f'\nAbsolute Error:')
        print(f'  Mean: {np.mean(absolute):.2f}ms')
        print(f'  Median: {np.median(absolute):.2f}ms')
        print(f'  Std: {np.std(absolute):.2f}ms')
        print(f'  Min/Max: {np.min(absolute):.2f}ms / {np.max(absolute):.2f}ms')


        print(f'\nR-value:')
        print(f'  Mean: {np.mean(r_value):.4f}')
        print(f'  Median: {np.median(r_value):.4f}')
        print(f'  90th percentile: {np.percentile(r_value, 90):.4f}')
        print(f'  95th percentile: {np.percentile(r_value, 95):.4f}')
        print(f'  Min/Max: {np.min(r_value):.4f} / {np.max(r_value):.4f}')

        print('\nR-value Thresholds:')
        for threshold in [1.5, 2.0, 3.0]:
            print(f'  R-value < {threshold}: {self._less_than_in_percent(r_value, threshold)}')

        print('\n' + '=' * 80)
        print('Detailed Results')
        print('=' * 80)

        table_data = []
        for r in results:
            table_data.append([
                r.label[:30],
                f'{r.predicted:.2f}',
                f'{r.measured.mean:.2f}',
                f'{r.vs_measured.absolute:.2f}',
                f'{r.vs_measured.r_value:.4f}',
            ])

        print(tabulate(
            table_data,
            headers=['Query', 'Estimated (ms)', 'Actual (ms)', 'Abs Error (ms)', 'R-value'],
            tablefmt='grid'
        ))
