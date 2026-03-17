from abc import ABC, abstractmethod
import datetime
import json
import random
from typing import Generic, TypeVar
from common.config import DatasetName
from common.drivers import DriverType
from common.utils import print_warning

TQuery = TypeVar('TQuery')

class Database(ABC, Generic[TQuery]):
    """
    Contains all dataset-specific logic for a single database in a dataset.
    """
    def __init__(self, dataset: DatasetName, type: DriverType):
        self.dataset = dataset
        self.type = type
        self.num_train_queries: int | None = None
        self.train_queries: list[TQuery] | None = None
        self.test_queries: dict[str, TestQuery[TQuery]] | None = None

    def id(self) -> str:
        """Get unique identifier for this database. Useful for caching."""
        return f'{self.dataset.value}_{self.type.value}'

    def get_train_queries(self, num_queries: int) -> list[TQuery]:
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

    def _train_query(self, query: TQuery):
        assert self.train_queries is not None
        self.train_queries.append(query)

    @abstractmethod
    def _generate_train_queries(self, num_queries: int):
        pass

    def get_test_queries(self) -> list['TestQuery[TQuery]']:
        """Generate a diverse set of test queries for evaluation. These should be different from training queries."""
        if self.test_queries is None:
            self.test_queries = {}
            self._generate_test_queries()

        return list(self.test_queries.values())

    def try_get_test_query(self, id: str) -> 'TestQuery[TQuery] | None':
        """Returns a single test query by its ID (or None if not found). Useful for debugging specific queries."""
        if self.test_queries is None:
            self.test_queries = {}
            self._generate_test_queries()

        return self.test_queries.get(id)

    def _test_query(self, id: str, name: str | None, content: TQuery):
        assert self.test_queries is not None
        self.test_queries[id] = TestQuery(id, name, content)

    @abstractmethod
    def _generate_test_queries(self):
        pass

    def _random_date(self, start_year=1992, end_year=1998) -> datetime.datetime:
        year = random.randint(start_year, end_year)
        month = random.randint(1, 12)
        day = random.randint(1, self.__days_in_month(month, year))
        return datetime.datetime(year, month, day)

    def _random_date_string(self, start_year=1992, end_year=1998) -> str:
        return self._random_date(start_year, end_year).strftime('%Y-%m-%d')

    def __days_in_month(self, month: int, year: int) -> int:
        if month == 2:
            return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
        elif month in [4, 6, 9, 11]:
            return 30
        else:
            return 31

class TestQuery(Generic[TQuery]):
    def __init__(self, id: str, name: str | None, content: TQuery):
        self.id = id
        self.name = name
        self.content = content

    def label(self) -> str:
        name = self.name if self.name else '(no name)'
        return f'{self.id}: {name}'

    # TODO Add some method to get content as string for printing

class MongoFindQuery:
    def __init__(self,
        collection: str,
        filter: dict | None = None,
        projection: dict | None = None,
        sort: dict | None = None,
        limit: int | None = None,
        skip: int | None = None,
    ):
        self.collection = collection
        self.filter = filter
        self.projection = projection
        self.sort = sort
        self.limit = limit
        self.skip = skip

    def __str__(self) -> str:
        return json.dumps({
            'collection': self.collection,
            'filter': self.filter,
            'projection': self.projection,
            'sort': self.sort,
            'limit': self.limit,
            'skip': self.skip
        }, indent=4)

class MongoAggregateQuery:
    def __init__(self, collection: str, pipeline: list[dict]):
        self.collection = collection
        self.pipeline = pipeline

    def __str__(self) -> str:
        return json.dumps({
            'collection': self.collection,
            'pipeline': self.pipeline
        }, indent=4)

MongoQuery = MongoFindQuery | MongoAggregateQuery

def try_parse_mongo_query(query_str: str) -> MongoQuery | None:
    """Parse a JSON string into a MongoQuery object."""
    try:
        query_dict = json.loads(query_str)
        if 'pipeline' in query_dict:
            return MongoAggregateQuery(collection=query_dict['collection'], pipeline=query_dict['pipeline'])
        else:
            return MongoFindQuery(
                collection=query_dict['collection'],
                filter=query_dict.get('filter'),
                projection=query_dict.get('projection'),
                sort=query_dict.get('sort'),
                limit=query_dict.get('limit'),
                skip=query_dict.get('skip'),
            )
    except (json.JSONDecodeError, KeyError) as e:
        print_warning(f'Invalid query format', e)
        return None
