from typing_extensions import override
from pymongo import ASCENDING
from common.drivers import MongoDriver
from common.daos.base_dao import BaseDAO

class MongoDAO(BaseDAO):
    def __init__(self, driver: MongoDriver):
        self.driver = driver
        self._db = driver.database()

    @override
    def find(self, entity: str, query_params):
        collection = self._db[entity]
        mongo_query = {}
        for key, value in query_params.items():
            if key.endswith('__in'):
                mongo_query[key[:-4]] = {'$in': value}
            else:
                mongo_query[key] = value

        return list(collection.find(mongo_query, {'_id': 0}))

    @override
    def insert(self, entity: str, data: dict):
        collection = self._db[entity]
        collection.insert_one(data)

    @override
    def create_kind_schema(self, entity: str, schema: list[dict]):
        pk_cols = [col['name'] for col in schema if col.get('primary_key')]
        if pk_cols:
            collection = self._db[entity]
            index_keys = [(key, ASCENDING) for key in pk_cols]

            try:
                collection.create_index(index_keys, unique=True)
                print(f'Created unique index on {pk_cols} for collection "{entity}"')
            except Exception as e:
                print(f'Could not create index on {entity}: {e}')

        print(f'Collection "{entity}" is ready in MongoDB.')

    @override
    def drop_kinds(self, populate_order: list[str]) -> None:
        for entity in reversed(populate_order):
            try:
                self._db.drop_collection(entity)
                print(f'Collection "{entity}" has been dropped in MongoDB.')
            except Exception as e:
                print(f'Skipping delete for {entity}: {e}')

    @override
    def reset_database(self) -> None:
        collection_names = self._db.list_collection_names()
        self.drop_kinds(collection_names)
