
from abc import ABC, abstractmethod

class Database(ABC):
    """
    Contains all dataset-specific logic for a single database in a dataset.
    """
    def __init__(self):
        self.num_train_queries: int | None = None
        self.train_queries: list[str] | None = None
        self.test_queries: dict[str, TestQuery] | None = None

    @abstractmethod
    def id(self) -> str:
        """Get unique identifier for this database. Useful for caching."""
        pass

    def get_train_queries(self, num_queries: int) -> list[str]:
        """
        Generate queries with parameter variations.
        Args:
            num_queries: Number of queries to generate
        Returns:
            List of SQL query strings
        """
        if self.train_queries is None or self.num_train_queries != num_queries:
            self.train_queries = []
            self.num_train_queries = num_queries
            self._generate_train_queries(num_queries)
            self.train_queries = self.train_queries[:num_queries]

        return self.train_queries

    def _train_query(self, query: str):
        assert self.train_queries is not None
        self.train_queries.append(query)

    @abstractmethod
    def _generate_train_queries(self, num_queries: int):
        pass

    def get_test_queries(self) -> list['TestQuery']:
        """Generate a diverse set of test queries for evaluation. These should be different from training queries."""
        if self.test_queries is None:
            self.test_queries = {}
            self._generate_test_queries()

        return list(self.test_queries.values())

    def get_test_query(self, id: str) -> 'TestQuery':
        """Returns a single test query by its ID (or None if not found). Useful for debugging specific queries."""
        if self.test_queries is None:
            self.test_queries = {}
            self._generate_test_queries()

        query = self.test_queries.get(id)
        if query is None:
            raise ValueError(f'Test query with ID {id} not found. Available IDs: {list(self.test_queries.keys())}')

        return query

    def _test_query(self, id: str, name: str | None, content: str):
        assert self.test_queries is not None
        self.test_queries[id] = TestQuery(id, name, content)

    @abstractmethod
    def _generate_test_queries(self):
        pass

class TestQuery:
    def __init__(self, id: str, name: str | None, content: str):
        self.id = id
        self.name = name
        self.content = content

    def label(self) -> str:
        name = self.name if self.name else '(no name)'
        return f'{self.id}: {name}'
