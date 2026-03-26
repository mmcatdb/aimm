from datetime import datetime, timedelta
from enum import Enum
import json
from typing import Any, TypeVar
from common.config import DatasetName
from common.drivers import DriverType
from common.utils import print_warning
from common.query_registry import QueryRegistry, TQuery

class DatabaseInfo:
    """Identifies a specific database."""

    def __init__(self, dataset: DatasetName, type: DriverType):
        self.dataset = dataset
        self.type = type

    def id(self) -> str:
        """Get unique identifier for this database. Useful for caching."""
        return f'{self.dataset.value}_{self.type.value}'

    def label(self) -> str:
        """Get a human-readable label for this database."""
        return f'{self.dataset.label()} ({self.type.value.capitalize()})'

class ValueType(Enum):
    STRING = 'string'
    NUMBER = 'number'
    DATE = 'date'

TListItem = TypeVar('TListItem')

class Database(QueryRegistry[TQuery], DatabaseInfo):
    """Contains all dataset-specific logic for a single database in a dataset."""

    def __init__(self, dataset: DatasetName, type: DriverType):
        DatabaseInfo.__init__(self, dataset, type)
        QueryRegistry.__init__(self)

    def _convert_scalar(self, value: Any, type: ValueType) -> Any:
        """Converts a scalar value to a representation suitable for queries."""
        if type == ValueType.STRING:
            return self._convert_string(value)
        elif type == ValueType.NUMBER:
            return str(value)
        elif type == ValueType.DATE:
            return self._convert_date(value)

    def _convert_string(self, value: str) -> Any:
        if self.type == DriverType.POSTGRES or self.type == DriverType.NEO4J:
            return f"'{value}'"

        # MongoDB doesn't need any conversion.
        return value

    def _convert_date(self, date: datetime) -> Any:
        """Converts a date to a representation suitable for queries."""
        if self.type == DriverType.POSTGRES or self.type == DriverType.NEO4J:
            return date.strftime('%Y-%m-%d')

        # MongoDB doesn't need any conversion.
        return date

    def _convert_array(self, array: list[Any], type: ValueType) -> Any:
        """Converts an array of values to a representation suitable for queries."""
        if self.type == DriverType.POSTGRES or self.type == DriverType.NEO4J:
            # The queries are supposed to put the correct brackets around.
            return ', '.join(map(lambda v: self._convert_scalar(v, type), array))

        # MongoDB doesn't need any conversion.
        return array

    def _rng_date(self, start_year=1992, end_year=1998) -> datetime:
        years = end_year - start_year + 1
        # Not the most accurate way to handle leap years, but good enough for random generation.
        seconds = years * 365 * 24 * 60 * 60
        return datetime(start_year, 1, 1) + timedelta(seconds = self._rng.randint(0, seconds))

    # Common utility methods for generating random parameters. Could be overridden by specific databases if needed.

    def _param_month(self):
        return self._param_int('month', 1, 12)

    def _rng_int(self, min_value: int, max_value: int):
        return self._rng.randint(min_value, max_value)

    def _param_int(self, name: str, min_value: int, max_value: int):
        return self._param(name, lambda: self._rng_int(min_value, max_value))

    def _param_float(self, name: str, min_value: float, max_value: float):
        return self._param(name, lambda: self._rng.uniform(min_value, max_value))

    def _param_choice(self, name: str, choices: list[TListItem]):
        return self._param(name, lambda: self._rng.choice(choices))

    def _param_limit(self, min_value: int = 10, max_order: int = 5):
        """Returns a random choice from [min_value * 2^n] for n in [0, ..., max_order]."""
        choices: list[int] = [min_value * (2 ** n) for n in range(max_order + 1)]
        return self._param_choice('limit', choices)

    def _param_skip(self, min_value: int = 10, max_value: int = 1000):
        return self._param_int('skip', min_value, max_value)

    def _param_int_array(self, name: str, max_value: int, min_count: int, max_count: int | None):
        """Useful for ids with the IN operator."""
        if max_count is None:
            max_count = min_count

        return self._param(name, lambda: self._convert_array([
            self._rng_int(1, max_value) for _ in range(self._rng_int(min_count, max_count))
        ], ValueType.NUMBER))


class MongoFindQuery:
    def __init__(self,
        collection: str,
        filter: dict | None = None,
        projection: dict | None = None,
        sort: dict | None = None,
        limit: int | str | None = None,
        skip: int | str | None = None,
    ):
        self.collection = collection
        self.filter = filter
        self.projection = projection
        self.sort = sort
        # We allow placeholder values but keep them separate.
        self._limit = limit
        self.limit = limit if isinstance(limit, int) else None
        self._skip = skip
        self.skip = skip if isinstance(skip, int) else None

    def __str__(self) -> str:
        return json.dumps({
            'collection': self.collection,
            'filter': self.filter,
            'projection': self.projection,
            'sort': self.sort,
            'limit': self._limit,
            'skip': self._skip
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
