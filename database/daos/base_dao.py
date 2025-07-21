from abc import ABC, abstractmethod


class BaseDAO(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def find(self, entity_name, query_params):
        pass

    @abstractmethod
    def insert(self, entity_name, data):
        pass

    @abstractmethod
    def create_schema(self, entity_name, schema):
        pass

    @abstractmethod
    def delete_all_from(self, entity_name):
        pass

    @abstractmethod
    def drop_entity(self, entity_name):
        pass