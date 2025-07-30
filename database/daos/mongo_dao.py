from pymongo import MongoClient, ASCENDING
from .base_dao import BaseDAO

class MongoDAO(BaseDAO):
    def __init__(self, config):
        self.config = config
        self.client = None
        self.db = None
        if config:
            self.connect()

    def connect(self):
        self.client = MongoClient(self.config['uri'])
        self.db = self.client[self.config['database']]

    def disconnect(self):
        if self.client:
            self.client.close()

    def find(self, entity_name, query_params):
        collection = self.db[entity_name]
        mongo_query = {}
        for key, value in query_params.items():
            if key.endswith("__in"):
                mongo_query[key[:-4]] = {"$in": value}
            else:
                mongo_query[key] = value
                
        return list(collection.find(mongo_query, {'_id': 0}))

    def insert(self, entity_name, data):
        collection = self.db[entity_name]
        collection.insert_one(data)

    def create_schema(self, entity_name, schema):
        pk_cols = [col['name'] for col in schema if col.get('primary_key')]
        if pk_cols:
            collection = self.db[entity_name]
            index_keys = [(key, ASCENDING) for key in pk_cols]
            
            try:
                collection.create_index(index_keys, unique=True)
                print(f"Created unique index on {pk_cols} for collection '{entity_name}'")
            except Exception as e:
                print(f"Could not create index on {entity_name}: {e}")
                
        print(f"Collection '{entity_name}' is ready in MongoDB.")


    def delete_all_from(self, entity_name):
        self.db[entity_name].delete_many({})
        print(f"All data from '{entity_name}' has been deleted in MongoDB.")

    def drop_entity(self, entity_name):
        self.db.drop_collection(entity_name)
        print(f"Collection '{entity_name}' has been dropped in MongoDB.")

    # A1) Non-Indexed Columns
    def get_all_lineitems(self):
        return list(self.db.lineitem.find({}, {'_id': 0}))

    # A2) Non-Indexed Columns - Range Query
    def get_orders_by_daterange(self, start_date, end_date):
        query = {
            "o_orderdate": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        return list(self.db.orders.find(query, {'_id': 0}))

    # A3) Indexed Columns
    def get_all_customers(self):
        return list(self.db.customer.find({}, {'_id': 0}))

    # A4) Indexed Columns - Range Query
    def get_orders_by_keyrange(self, start_key, end_key):
        query = {
            "o_orderkey": {
                "$gte": start_key,
                "$lte": end_key
            }
        }
        return list(self.db.orders.find(query, {'_id': 0}))

    # B1) COUNT
    def count_orders_by_month(self):
        pipeline = [
            {
                "$group": {
                    "_id": { "$substr": ["$o_orderdate", 0, 7] },
                    "order_count": { "$sum": 1 }
                }
            },
            {
                "$project": {
                    "order_month": "$_id",
                    "order_count": 1,
                    "_id": 0
                }
            },
            { "$sort": { "order_month": 1 } }
        ]
        
        return list(self.db.orders.aggregate(pipeline))

    # B2) MAX
    def get_max_price_by_ship_month(self):
        pipeline = [
            {
                "$group": {
                    "_id": { "$substr": ["$l_shipdate", 0, 7] },
                    "max_price": { "$max": "$l_extendedprice" }
                }
            },
            {
                "$project": {
                    "ship_month": "$_id",
                    "max_price": 1,
                    "_id": 0
                }
            },
            { "$sort": { "ship_month": 1 } }
        ]
        return list(self.db.lineitem.aggregate(pipeline))
