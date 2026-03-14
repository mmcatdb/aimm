from typing_extensions import override
from pymongo import ASCENDING
from common.drivers import MongoDriver
from common.daos.base_dao import BaseDAO

class IndexSchema:
    def __init__(self, kind: str, keys: list[str], is_unique=False):
        """The keys can be nested, e.g., `user.address.street`."""
        self.kind = kind
        self.keys = keys
        self.is_unique = is_unique

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

    def create_index(self, index: IndexSchema):
        """The keys can be nested, e.g., `user.address.street`."""
        collection = self._db[index.kind]
        # The direction shouldn't matter. ASCENDING is the default.
        keys = [(key, ASCENDING) for key in index.keys]

        unique = 'unique ' if index.is_unique else ''
        message = f'{unique}index on "{index.keys}" for collection "{index.kind}"'

        try:
            collection.create_index(keys, unique=index.is_unique)
            print(f'Created {message}')
        except Exception as e:
            print(f'Could not create {message}: {e}')

    @override
    def drop_kinds(self, populate_order: list[str]):
        for entity in reversed(populate_order):
            try:
                self._db.drop_collection(entity)
                print(f'Collection "{entity}" has been dropped in MongoDB.')
            except Exception as e:
                print(f'Skipping delete for "{entity}": {e}')

    @override
    def reset_database(self):
        collection_names = self._db.list_collection_names()
        self.drop_kinds(collection_names)
