from abc import abstractmethod
import re
from typing import Generic
from core.drivers import DriverType
from core.query import ABC, MongoAggregateQuery, MongoFindQuery, MongoQuery, TQuery, override

# If more driver-specific logic is needed, we should extend these classes ... probably also rename them to something more generic than "KindExtractor" and move them to separate files.

class KindExtractor(ABC, Generic[TQuery]):

    @abstractmethod
    def extract_query_kinds(self, query: TQuery) -> set[str]:
        """Returns the set of kinds that have to be in the database for the query to be executed."""
        pass

def get_kind_extractor(driver_type: DriverType) -> KindExtractor:
    if driver_type == DriverType.POSTGRES:
        return PostgresExtractor()
    elif driver_type == DriverType.MONGO:
        return MongoExtractor()
    elif driver_type == DriverType.NEO4J:
        return Neo4jExtractor()

#region Postgres

class PostgresExtractor(KindExtractor[str]):

    @override
    def extract_query_kinds(self, query: str) -> set[str]:
        """Returns the set of kinds that have to be in the database for the query to be executed."""

        # TODO Check this ...

        ctes = {
            *CTE_PATTERN.findall(query),
            *CTE_FOLLOWING_PATTERN.findall(query),
        }
        ctes_lower = {cte.lower() for cte in ctes}

        tables = set[str]()
        for match in SOURCE_PATTERN.finditer(query):
            first_quoted, first_plain, second_quoted, second_plain = match.groups()

            first_part = first_quoted or first_plain
            second_part = second_quoted or second_plain
            table_name = second_part if second_part else first_part

            if not table_name:
                continue
            if table_name.lower() in ctes_lower:
                continue
            tables.add(table_name)

        return tables

CTE_PATTERN = re.compile(r'\bWITH\s+([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(', re.IGNORECASE)
CTE_FOLLOWING_PATTERN = re.compile(r',\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(', re.IGNORECASE)
SOURCE_PATTERN = re.compile(r'\b(?:FROM|JOIN)\s+(?:ONLY\s+)?(?:(?:"([^"]+)")|([A-Za-z_][A-Za-z0-9_]*))(?:\s*\.\s*(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*)))?', re.IGNORECASE)

#endregion
#region Mongo

class MongoExtractor(KindExtractor[MongoQuery]):

    def extract_query_kinds(self, query: MongoQuery) -> set[str]:
        collections = set[str]()

        if isinstance(query, MongoFindQuery):
            collections.add(query.collection)
        elif isinstance(query, MongoAggregateQuery):
            collections.add(query.collection)
            collections.update(self.__extract_mongo_pipeline_collections(query.pipeline))

        return collections

    def __extract_mongo_pipeline_collections(self, pipeline: list[dict]) -> set[str]:
        collections = set[str]()

        def append_collection(collection: str | None):
            if collection is not None:
                collections.add(collection)

        def visit_node(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ('$lookup', '$graphLookup') and isinstance(value, dict):
                        append_collection(value.get('from'))
                        if 'pipeline' in value:
                            visit_node(value['pipeline'])
                        continue

                    if key == '$unionWith':
                        if isinstance(value, str):
                            append_collection(value)
                        elif isinstance(value, dict):
                            append_collection(value.get('coll'))
                            if 'pipeline' in value:
                                visit_node(value['pipeline'])
                        continue

                    visit_node(value)
            elif isinstance(node, list):
                for item in node:
                    visit_node(item)

        visit_node(pipeline)

        return collections

#endregion
#region Neo4j

class Neo4jExtractor(KindExtractor[str]):

    def extract_query_kinds(self, query: str) -> set[str]:
        labels = NODE_PATTERN.findall(query)
        edges = EDGE_PATTERN.findall(query)

        return set(labels + edges)

NODE_PATTERN = re.compile(r'\(\s*\w*\s*:\s*([A-Za-z0-9_]+)')
# FIXME Check this ...
EDGE_PATTERN = re.compile(r'\[\s*\w*\s*:\s*(FOLLOWS|HAS_CATEGORY)\b')

#endregion
