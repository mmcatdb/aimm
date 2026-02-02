from typing_extensions import override
from pymongo import ASCENDING
from common.drivers import MongoDriver
from common.daos.base_dao import BaseDAO

class MongoDAO(BaseDAO):
    def __init__(self, driver: MongoDriver):
        self.driver = driver
        self._db = driver.database()

    @override
    def find(self, entity_name, query_params):
        collection = self._db[entity_name]
        mongo_query = {}
        for key, value in query_params.items():
            if key.endswith('__in'):
                mongo_query[key[:-4]] = {'$in': value}
            else:
                mongo_query[key] = value

        return list(collection.find(mongo_query, {'_id': 0}))

    @override
    def insert(self, entity_name, data):
        collection = self._db[entity_name]
        collection.insert_one(data)

    @override
    def create_schema(self, entity_name, schema):
        pk_cols = [col['name'] for col in schema if col.get('primary_key')]
        if pk_cols:
            collection = self._db[entity_name]
            index_keys = [(key, ASCENDING) for key in pk_cols]

            try:
                collection.create_index(index_keys, unique=True)
                print(f'Created unique index on {pk_cols} for collection "{entity_name}"')
            except Exception as e:
                print(f'Could not create index on {entity_name}: {e}')

        print(f'Collection "{entity_name}" is ready in MongoDB.')

    @override
    def delete_all_from(self, entity_name):
        self._db[entity_name].delete_many({})
        print(f'All data from "{entity_name}" has been deleted in MongoDB.')

    @override
    def drop_entity(self, entity_name):
        self._db.drop_collection(entity_name)
        print(f'Collection "{entity_name}" has been dropped in MongoDB.')
