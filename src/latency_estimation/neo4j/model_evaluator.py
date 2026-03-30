from math import nan
import torch
import numpy as np
from tabulate import tabulate
from dataclasses import dataclass
from common.utils import print_warning
from common.query_registry import QueryDef
from latency_estimation.neo4j.plan_structured_network import PlanStructuredNetwork
from latency_estimation.neo4j.plan_extractor import PlanExtractor

class ModelEvaluator:
    """Evaluates a trained Neo4j QPP model."""
    def __init__(self, extractor: PlanExtractor, model: PlanStructuredNetwork):
        self.extractor = extractor
        self.model = model

    def evaluate_multiple_queries(self, queries: list[QueryDef[str]], num_runs: int) -> list['Result']:
        results = list['Result']()

        print(f'\nEvaluating {len(queries)} queries...')
        print('=' * 80)

        for query in queries:
            try:
                result = self.evaluate_query(query, num_runs)
                results.append(result)
            except Exception as e:
                print_warning(f'Could not evaluate query {query.id}: {query.label()}.', e)
                continue

        return results

    def evaluate_query(self, query: QueryDef[str], num_runs: int) -> 'Result':
        print(f'\nEvaluating: {query.label()}')

        result = Result(query.label())

        content = query.generate()
        plan = self.extractor.explain_query(content)
        predicted_ms = self.__estimate_latency(plan)

        times = self.extractor.measure_query_multiple(content, num_runs)

        actual_ms = np.mean(times).item()
        result.std_latency = np.std(times).item()

        result.predicted_ms = predicted_ms
        result.actual_ms = actual_ms

        # Compute metrics
        result.error_ms = abs(predicted_ms - actual_ms)
        result.r_value = max(predicted_ms / actual_ms, actual_ms / predicted_ms) \
            if predicted_ms > 0 and actual_ms > 0 else float('inf')

        print(f'  Estimated: {predicted_ms:.2f}ms')
        print(f'  Actual: {actual_ms:.2f}ms (±{result.std_latency:.2f}ms)')
        print(f'  Absolute Error: {result.error_ms:.2f}ms')
        print(f'  R-value: {result.r_value:.4f}')

        return result

    def __estimate_latency(self, plan: dict) -> float:
        with torch.no_grad():
            return self.model(plan).item()

    def print_summary(self, results: list['Result']):
        """Print summary statistics of evaluation results."""
        if not results:
            print('\nNo results to summarize.')
            return

        print('\n' + '=' * 80)
        print('Evaluation Summary')
        print('=' * 80)

        # Extract metrics
        errors = [r.error_ms for r in results]
        r_values = [r.r_value for r in results if r.r_value != float('inf')]

        # Compute statistics
        print(f'\nNumber of queries: {len(results)}')
        print(f'\nAbsolute Error:')
        print(f'  Mean: {np.mean(errors):.2f}ms')
        print(f'  Median: {np.median(errors):.2f}ms')
        print(f'  Std: {np.std(errors):.2f}ms')
        print(f'  Min/Max: {np.min(errors):.2f}ms / {np.max(errors):.2f}ms')


        if r_values:
            print(f'\nR-value:')
            print(f'  Mean: {np.mean(r_values):.4f}')
            print(f'  Median: {np.median(r_values):.4f}')
            print(f'  90th percentile: {np.percentile(r_values, 90):.4f}')
            print(f'  95th percentile: {np.percentile(r_values, 95):.4f}')
            print(f'  Min/Max: {np.min(r_values):.4f} / {np.max(r_values):.4f}')

            r_value_thresholds = [1.5, 2.0, 3.0]
            total_queries = len(results)
            print('\nR-value Thresholds:')
            for threshold in r_value_thresholds:
                count_below = sum(1 for r in results if r.r_value < threshold)
                percent_below = (count_below / total_queries) * 100
                print(f'  R-value < {threshold}: {count_below} queries ({percent_below:.2f}%)')

        print('\n' + '=' * 80)
        print('Detailed Results')
        print('=' * 80)

        table_data = []
        for r in results:
            table_data.append([
                r.label[:30],
                f'{r.predicted_ms:.2f}',
                f'{r.actual_ms:.2f}',
                f'{r.error_ms:.2f}',
                f'{r.r_value:.4f}' if r.r_value != float('inf') else 'inf'
            ])

        print(tabulate(
            table_data,
            headers=['Query', 'Estimated (ms)', 'Actual (ms)', 'Abs Error (ms)', 'R-value'],
            tablefmt='grid'
        ))

@dataclass
class Result:
    """Holds a single query evaluation result."""
    label: str
    predicted_ms: float = nan
    actual_ms: float = nan
    std_latency: float = nan
    error_ms: float = nan
    r_value: float = nan
