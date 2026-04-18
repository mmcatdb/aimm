from typing import Generic, Literal, TypeVar, overload
from core.drivers import DriverType
from .query_id import QueryInstanceId, SchemaName, create_database_id, create_query_instance_id
from .mongo_query import MongoQuery

TQuery = TypeVar('TQuery', str, MongoQuery)
TQuery_cov = TypeVar('TQuery_cov', str, MongoQuery, covariant=True)

class QueryInstance(Generic[TQuery]):
    """An instantiated `QueryTemplate` with specific parameters."""

    def __init__(self, id: QueryInstanceId, label: str, content: TQuery):
        self.id = id
        self.label = label
        self.content = content

    @staticmethod
    def create_custom(driver: DriverType, schema: SchemaName, scale: float, index: int, content: TQuery):
        """Creates a QueryInstance with given content. Useful for testing and debugging."""
        database_id = create_database_id(driver, schema, scale)
        id = create_query_instance_id(database_id, 'custom', index)
        return QueryInstance(id, f'Custom Query {index}', content)

@overload
def parse_query(driver_type: Literal[DriverType.MONGO], content_str: str) -> MongoQuery: ...

@overload
def parse_query(driver_type: Literal[DriverType.POSTGRES, DriverType.NEO4J], content_str: str) -> str: ...

def parse_query(driver_type: DriverType, content_str: str) -> str | MongoQuery:
    if driver_type == DriverType.MONGO:
        return MongoQuery.parse(content_str)
    return content_str
