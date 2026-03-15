from typing import Generic, TypeVar
from sklearn.externals.array_api_compat.numpy import ndarray
from torch.utils.data import Dataset

TQuery = TypeVar('TQuery')

class BaseDataset(Dataset, Generic[TQuery]):
    """
    Dataset of query plans with execution times.

    Each item contains:
    - query: The Cypher query string
    - plan: The query execution plan (from EXPLAIN)
    - execution_time: Actual measured execution time in seconds
    """
    def __init__(self, queries: list[TQuery], plans: list[dict], execution_times: list[float],
    ):
        assert len(queries) == len(plans) == len(execution_times), 'Queries, plans, and execution times must have same length'

        self.queries = queries
        self.plans = plans
        self.execution_times = execution_times

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return {
            'query': self.queries[index],
            'plan': self.plans[index],
            'execution_time': self.execution_times[index]
        }

    def subset(self, indexes: list[int] | ndarray) -> 'BaseDataset':
        """Create a subset of the dataset based on given indexes."""
        return BaseDataset(
            queries = [self.queries[i] for i in indexes],
            plans = [self.plans[i] for i in indexes],
            execution_times = [self.execution_times[i] for i in indexes],
        )
