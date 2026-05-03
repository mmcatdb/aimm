from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from core.utils import EPSILON, print_warning
from latency_estimation.dataset import ArrayDataset, DatasetItem
from latency_estimation.model import BaseModel

if TYPE_CHECKING:
    from matplotlib.figure import Figure

@dataclass
class DiffResult:
    absolute: float
    relative: float
    r_value: float
    ratio: float

@dataclass
class QueryResult:
    id: str
    label: str
    predicted: float

    measured: float
    vs_measured: DiffResult

@dataclass
class ExtractedQueryResult(QueryResult):
    extracted: float
    vs_extracted: DiffResult
    extracted_vs_measured: DiffResult

class BaseModelEvaluator(ABC):
    """Evaluates a trained QPP-Net model on a dataset."""

    def __init__(self, estimator: BaseModel):
        self.estimator = estimator

    def evaluate_dataset(self, dataset: ArrayDataset) -> list[QueryResult]:
        results = list[QueryResult]()

        print('=' * 80)
        print(f'EVALUATING {len(dataset)} QUERIES')
        print('=' * 80)

        for item in dataset:
            try:
                result = self._evaluate_dataset_item(item)
                results.append(result)
            except Exception as e:
                print_warning(f'Could not evaluate query {item.query_id}: {item.label}.', e)

        return results

    def _evaluate_dataset_item(self, item: DatasetItem) -> QueryResult:
        predicted = self.estimator.evaluate(item.plan)

        measured = item.latency
        vs_measured = self._compute_diff_result(predicted, measured)

        base_result = QueryResult(
            id=item.query_id,
            label=item.label,
            predicted=predicted,
            measured=measured,
            vs_measured=vs_measured,
        )

        extracted = item.plan.latency
        if extracted is None:
            return base_result

        vs_extracted = self._compute_diff_result(predicted, extracted)
        extracted_vs_measured = self._compute_diff_result(extracted, measured)

        return ExtractedQueryResult(
            **vars(base_result),
            extracted=extracted,
            vs_extracted=vs_extracted,
            extracted_vs_measured=extracted_vs_measured,
        )

    @staticmethod
    def _get_extracted_results(results: list[QueryResult]) -> list[ExtractedQueryResult]:
        extracted_results = list[ExtractedQueryResult]()

        for r in results:
            if not isinstance(r, ExtractedQueryResult):
                print_warning(f'Query {r.label} does not have extracted latency. Skipping from summary.')
            else:
                extracted_results.append(r)

        return extracted_results

    @abstractmethod
    def print_summary(self, results: list[QueryResult]):
        """Print summary statistics and comparison table."""
        pass

    def _compute_diff_result(self, a: float, b: float) -> DiffResult:
        absolute = abs(a - b)
        relative = absolute / (b + EPSILON) if b > 0 else float('inf')
        r_value = max(a / (b + EPSILON), b / (a + EPSILON)) if a > 0 and b > 0 else float('inf')
        ratio = a / (b + EPSILON) if b > 0 else float('inf')
        return DiffResult(absolute, relative, r_value, ratio)

    def _summary_diff_results(self, results: list[DiffResult], r_value_thresholds: list[float] | None = None):
        absolute = [r.absolute for r in results]
        relative = [r.relative for r in results]

        print(f'  Mean Absolute Error: {np.mean(absolute):.2f} ms')
        print(f'  Median Absolute Error: {np.median(absolute):.2f} ms')
        print(f'  Mean Relative Error: {np.mean(relative):.4f}')

        if r_value_thresholds is not None:
            r_value = [r.r_value for r in results]
            print(f'  Median R-value: {np.median(r_value):.2f}')

            for threshold in r_value_thresholds:
                print(f'  R < {threshold}: {self._less_than_in_percent(r_value, threshold)}')

    @staticmethod
    def _less_than_in_percent(values: list[float], threshold: float) -> str:
        ratio = np.mean([value < threshold for value in values]).item() * 100
        return f'{ratio:.1f} %'

    def plot_results(self, results: list[QueryResult], save_path: str) -> 'Figure | None':
        """Create visualization plots comparing estimations and actual times."""
        raise NotImplementedError('Plotting not implemented for this evaluator.')
