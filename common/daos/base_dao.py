from abc import ABC, abstractmethod
from typing import Any

class BaseDAO(ABC):
    @abstractmethod
    def find(self, entity_name: str, query_params) -> Any:
        """
        Handles simple conditional queries
        Supports exact matches (key1 = val1 AND key2 = val2...), and IN clauses (key__in = [v1, v2,...])
        """
        pass

    @abstractmethod
    def insert(self, entity_name, data) -> None:
        pass

    @abstractmethod
    def create_schema(self, entity_name, schema) -> None:
        pass

    @abstractmethod
    def delete_all_from(self, entity_name) -> None:
        pass

    @abstractmethod
    def drop_entity(self, entity_name) -> None:
        pass
