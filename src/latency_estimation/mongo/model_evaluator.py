from math import nan
import torch
import numpy as np
from dataclasses import dataclass
from common.utils import EPSILON
from common.database import TestQuery, MongoQuery, MongoFindQuery, MongoAggregateQuery
from latency_estimation.mongo.plan_extractor import PlanExtractor
from latency_estimation.mongo.plan_structured_network import PlanStructuredNetwork

class ModelEvaluator:
    """Evaluates a trained MongoDB QPP-Net model."""

    def __init__(self, extractor: PlanExtractor, model: PlanStructuredNetwork):
        self.extractor = extractor
        self.model = model

    def evaluate_multiple_queries(self, queries: list[TestQuery[MongoQuery]], num_runs: int) -> list['Result']:
        results: list['Result'] = []

        print('=' * 80)
        print(f'EVALUATING {len(queries)} QUERIES')
        print('=' * 80)

        for test_query in queries:
            query = test_query.content
            try:
                if isinstance(query, MongoFindQuery):
                    r = self.__evaluate_find(query, num_runs=num_runs, label=test_query.label())
                else:
                    r = self.__evaluate_aggregate(query, num_runs=num_runs, label=test_query.label())

                results.append(r)
                status = 'OK' if r.r_value <= 2.0 else ('WARN' if r.r_value <= 5.0 else 'BAD')
                print(
                    f'  [{status:4s}] {r.label[:55]:55s}'
                    f'  pred={r.predicted_ms:8.1f}ms  actual={r.actual_ms:8.1f}ms'
                    f'  R={r.r_value:.2f}'
                )

            except Exception as e:
                print(f'  [ERR ] {test_query.label()[:55]:55s}  {e}')

        return results

    def __evaluate_find(self, query: MongoFindQuery, num_runs: int, label: str) -> 'Result':
        """Evaluate a single find query."""
        # 1. Get plan without executing
        plan = self.extractor.explain_find(query, verbosity='queryPlanner')
        predicted_ms = self.__estimate_latency(plan, query.collection)

        # 2. Measure actual execution time
        times = self.extractor.measure_find(query, num_runs)

        return self.__create_result(label, predicted_ms, times)

    def __evaluate_aggregate(self, query: MongoAggregateQuery, num_runs: int, label: str) -> 'Result':
        """Evaluate a single aggregate query."""
        plan = self.extractor.explain_aggregate(query, verbosity='queryPlanner')
        predicted_ms = self.__estimate_latency(plan, query.collection)

        times = self.extractor.measure_aggregate(query, num_runs)

        return self.__create_result(label, predicted_ms, times)

    def __estimate_latency(self, plan: dict, collection_name: str) -> float:
        """Predict latency from a queryPlanner explain (no execution)."""
        with torch.no_grad():
            return self.model(plan, collection_name).item()

    def __create_result(self, label: str, predicted_ms: float, times: list[float]) -> 'Result':
        result = Result(label)

        actual_ms = np.mean(times).item()
        r_value = max(predicted_ms / (actual_ms + EPSILON), actual_ms / (predicted_ms + EPSILON))

        result.predicted_ms = predicted_ms
        result.actual_ms = actual_ms
        result.actual_min = np.min(times)
        result.actual_max = np.max(times)
        result.error_ms = abs(predicted_ms - actual_ms)
        result.relative_error = abs(predicted_ms - actual_ms) / (actual_ms + EPSILON)
        result.r_value = r_value

        return result

    def print_summary(self, results: list['Result']):
        """Print summary statistics and comparison table."""
        print('\n' + '=' * 80)
        print('AGGREGATE STATISTICS')
        print('=' * 80)

        errors = [r.error_ms for r in results]
        rel_errors = [r.relative_error for r in results]
        r_values = [r.r_value for r in results]
        predicted = [r.predicted_ms for r in results]
        actual = [r.actual_ms for r in results]

        print(f'  Queries evaluated: {len(results)}')
        print(f'  Mean Absolute Error: {np.mean(errors):.2f} ms')
        print(f'  Median Absolute Error: {np.median(errors):.2f} ms')
        print(f'  Mean Relative Error: {np.mean(rel_errors):.4f}')
        print(f'  Median R-value (Q-error): {np.median(r_values):.2f}')
        print(f'  Mean R-value: {np.mean(r_values):.2f}')
        print(f'  R <= 1.5: {np.mean([r <= 1.5 for r in r_values]) * 100:.1f}%')
        print(f'  R <= 2.0: {np.mean([r <= 2.0 for r in r_values]) * 100:.1f}%')
        print(f'  R <= 5.0: {np.mean([r <= 5.0 for r in r_values]) * 100:.1f}%')

@dataclass
class Result:
    """Holds a single query evaluation result."""
    label: str

    predicted_ms: float = nan
    actual_ms: float = nan
    actual_min: float = nan
    actual_max: float = nan
    error_ms: float = nan
    relative_error: float = nan
    r_value: float = nan
