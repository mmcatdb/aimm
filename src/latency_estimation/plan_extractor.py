from abc import ABC, abstractmethod
from typing import Generic
from common.query_registry import TQuery
from common.query_registry import QueryDef

class BasePlanExtractor(ABC, Generic[TQuery]):

    @abstractmethod
    def measure_query(self, query: TQuery) -> tuple[float, int]:
        """Measures single query execution. Returns tuple of (time_ms, num_results)."""
        pass

    def measure_query_multiple(self, query: TQuery, num_runs: int) -> list[float]:
        """Measures the query multiple times and returns list of time_ms."""
        return self.__measure_queries([query] * num_runs)

    def measure_query_generated(self, query_def: QueryDef[TQuery], num_runs: int) -> list[float]:
        """Generates a new query each time and measures it. Returns list of time_ms."""
        return self.__measure_queries([query_def.generate() for _ in range(num_runs)])

    def __measure_queries(self, queries: list[TQuery]) -> list[float]:
        """Measures multiple queries and returns list of time_ms."""
        times = []
        for query in queries:
            time, _ = self.measure_query(query)
            times.append(time)

        return times
