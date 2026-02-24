from abc import ABC, abstractmethod
from typing import Any

class BaseDAO(ABC):
    @abstractmethod
    def find(self, entity: str, query_params) -> Any:
        """
        Handles simple conditional queries
        Supports exact matches (key1 = val1 AND key2 = val2...), and IN clauses (key__in = [v1, v2,...])
        """
        pass

    @abstractmethod
    def insert(self, entity: str, data: dict) -> None:
        pass

    @abstractmethod
    def create_kind_schema(self, entity: str, schema: list[dict]) -> None:
        pass

    @abstractmethod
    def drop_kinds(self, populate_order: list[str]) -> None:
        """Drops just the kinds (tables/collections/nodes/edges) specified in populate_order."""
        pass

    @abstractmethod
    def reset_database(self) -> None:
        """Resets the entire database, dropping all kinds."""
        pass
