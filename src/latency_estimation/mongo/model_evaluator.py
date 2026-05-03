from typing_extensions import override
import numpy as np
from tabulate import tabulate
from ..model_evaluator import BaseModelEvaluator, QueryResult

class ModelEvaluator(BaseModelEvaluator):

    @override
    def print_summary(self, results: list[QueryResult]):
        print('\n' + '=' * 80)
        print('AGGREGATE STATISTICS')
        print('=' * 80)

        absolute = [r.vs_measured.absolute for r in results]
        relative = [r.vs_measured.relative for r in results]
        r_value = [r.vs_measured.r_value for r in results]

        print(f'  Queries evaluated: {len(results)}')
        print(f'  Mean Absolute Error: {np.mean(absolute):.2f} ms')
        print(f'  Median Absolute Error: {np.median(absolute):.2f} ms')
        print(f'  Mean Relative Error: {np.mean(relative):.4f}')
        print(f'  Median R-value (Q-error): {np.median(r_value):.2f}')
        print(f'  Mean R-value: {np.mean(r_value):.2f}')

        for threshold in [1.5, 2.0, 5.0]:
            print(f'  R < {threshold}: {self._less_than_in_percent(r_value, threshold)}')

        table_data = []
        for r in results:
            table_data.append([
                r.label[:30],
                f'{r.predicted:.2f}',
                f'{r.measured:.2f}',
                f'{r.vs_measured.absolute:.2f}',
                f'{r.vs_measured.r_value:.4f}',
            ])

        print(tabulate(
            table_data,
            headers=['Query', 'Predicted (ms)', 'Actual (ms)', 'Abs Error (ms)', 'R-value'],
            tablefmt='grid'
        ))
