from __future__ import annotations
from abc import ABC, abstractmethod
from typing_extensions import override
import json
from bson import json_util

class MongoQuery(ABC):
    """Base class for MongoDB queries. We currently support only find and aggregate queries, but this can be extended as needed."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the query to a dictionary format suitable for MongoDB commands and serialization."""
        pass

    def __str__(self) -> str:
        return json_util.dumps(self.to_dict(), indent=4)

    def serialize(self) -> str:
        """Like `__str__`, but don't pretty-print."""
        return json_util.dumps(self.to_dict())

    @staticmethod
    def parse(query_str: str) -> MongoQuery:
        """Parse a JSON string into a MongoQuery object. Raises error if parsing fails."""
        try:
            query_dict = json_util.loads(query_str)
            if 'find' in query_dict:
                return MongoFindQuery.from_dict(query_dict)
            else:
                return MongoAggregateQuery.from_dict(query_dict)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f'Failed to parse Mongo query from string: {query_str}') from e

class MongoFindQuery(MongoQuery):
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

    @override
    def to_dict(self) -> dict:
        return _filter_none({
            'find': self.collection,
            'filter': self.filter,
            'projection': self.projection,
            'sort': self.sort,
            'limit': self._limit,
            'skip': self._skip
        })

    @staticmethod
    def from_dict(data: dict) -> MongoFindQuery:
        return MongoFindQuery(
            collection=data['find'],
            filter=data.get('filter'),
            projection=data.get('projection'),
            sort=data.get('sort'),
            limit=data.get('limit'),
            skip=data.get('skip')
        )

class MongoAggregateQuery(MongoQuery):
    def __init__(self, collection: str, pipeline: list[dict]):
        self.collection = collection
        self.pipeline = pipeline

    @override
    def to_dict(self) -> dict:
        return {
            'aggregate': self.collection,
            'pipeline': self.pipeline
        }

    @staticmethod
    def from_dict(data: dict) -> MongoAggregateQuery:
        return MongoAggregateQuery(
            collection=data['aggregate'],
            pipeline=data['pipeline']
        )

def _filter_none(data: dict) -> dict:
    return {key: value for key, value in data.items() if value is not None}
