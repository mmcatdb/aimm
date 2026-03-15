from math import nan
import torch
import numpy as np
from tabulate import tabulate
from dataclasses import dataclass
from common.database import TestQuery
from latency_estimation.neo4j.plan_structured_network import PlanStructuredNetwork
from latency_estimation.neo4j.plan_extractor import PlanExtractor

class ModelEvaluator:
    """Evaluates a trained Neo4j QPP model."""
    def __init__(self, extractor: PlanExtractor, model: PlanStructuredNetwork):
        self.extractor = extractor
        self.model = model

    def evaluate_multiple_queries(self, queries: list[TestQuery[str]], num_runs: int) -> list['Result']:
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
                print(f'  ✗ Error evaluating {query.label()}: {str(e)}')
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
                f'{r.estimated_latency * 1000:.2f}',
                f'{r.actual_latency * 1000:.2f}',
                f'{r.absolute_error * 1000:.2f}',
                f'{r.r_value:.4f}' if r.r_value != float('inf') else 'inf'
            ])

        print('\n' + '=' * 70)
        print('Detailed Results')
        print('=' * 70)
        print(tabulate(
            table_data,
            headers=['Query', 'Estimated (ms)', 'Actual (ms)', 'Abs Error (ms)', 'R-value'],
            tablefmt='grid'
        ))

        r_value_thresholds = [1.5, 2.0, 3.0]
        total_queries = len(results)
        print('\nR-value Thresholds:')
        for threshold in r_value_thresholds:
            count_below = sum(1 for r in results if r.r_value < threshold)
            percent_below = (count_below / total_queries) * 100
            print(f'  R-value < {threshold}: {count_below} queries ({percent_below:.2f}%)')

    def evaluate_query(self, query: TestQuery[str], num_runs: int) -> 'Result':
        """
        Evaluate a single query.

        Args:
            num_runs: Number of executions for averaging
        """
        print(f'\nEvaluating: {query.label()}')

        plan = self.extractor.explain_plan(query.content)
        result = Result(query.label(), query.content, plan)

        estimated_latency = self.__estimate_latency(result.plan)
        actual_latency, result.std_latency, _ = self.extractor.measure_query(query.content, num_runs)

        result.estimated_latency = estimated_latency
        result.actual_latency = actual_latency

        # Compute metrics
        result.absolute_error = abs(estimated_latency - actual_latency)
        result.r_value = max(estimated_latency / actual_latency, actual_latency / estimated_latency) \
            if estimated_latency > 0 and actual_latency > 0 else float('inf')

        print(f'  Estimated: {estimated_latency * 1000:.2f}ms')
        print(f'  Actual: {actual_latency * 1000:.2f}ms (±{result.std_latency * 1000:.2f}ms)')
        print(f'  Absolute Error: {result.absolute_error * 1000:.2f}ms')
        print(f'  R-value: {result.r_value:.4f}')

        return result

    def __estimate_latency(self, plan: dict) -> float:
        with torch.no_grad():
            return self.model(plan).item()

@dataclass
class Result:
    """Holds a single query evaluation result."""
    name: str
    content: str
    plan: dict

    estimated_latency: float = nan
    actual_latency: float = nan
    std_latency: float = nan
    absolute_error: float = nan
    r_value: float = nan
