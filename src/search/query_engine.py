from abc import ABC, abstractmethod

class QueryEngine(ABC):
    """Provides a unified interface to measure query execution time across different databases."""

    @abstractmethod
    def available_databases(self) -> list[str]:
        """Returns a list of available database ids, e.g. [ 'postgres', 'mongo', 'neo4j' ]."""
        pass

    @abstractmethod
    def measure_queries(self, mapping: dict[str, str], verbose=True) -> float:
        """
        Measures the execution time of a predefined set of queries given a mapping of tables to databases.
        Args:
            mapping: A mapping from table names to database ids, e.g. { 'customer': 'postgres', 'orders': 'neo4j', 'lineitem': 'mongo' }.
            verbose: Whether to print detailed statistics about each query execution.
        """
        pass

