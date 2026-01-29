from pymongo import ASCENDING

from common.databases import Mongo
from .base_dao import BaseDAO

class MongoDAO(BaseDAO):
    def __init__(self, mongo: Mongo):
        self.mongo = mongo
        self.db = mongo.database()

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

    def get_all_lineitems(self):
        return list(self.db.lineitem.find({}, {'_id': 0}))

    def get_orders_by_daterange(self, start_date, end_date):
        query = {
            "o_orderdate": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        return list(self.db.orders.find(query, {'_id': 0}))

    def get_all_customers(self):
        return list(self.db.customer.find({}, {'_id': 0}))

    def get_orders_by_keyrange(self, start_key, end_key):
        query = {
            "o_orderkey": {
                "$gte": start_key,
                "$lte": end_key
            }
        }
        return list(self.db.orders.find(query, {'_id': 0}))

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

    # --- Part / Supplier / PartSupp ---
    def get_all_parts(self):
        return list(self.db.part.find({}, {'_id': 0}))

    def get_parts_by_size_range(self, min_size, max_size):
        query = {"p_size": {"$gte": str(min_size), "$lte": str(max_size)}}
        return list(self.db.part.find(query, {'_id': 0}))

    def get_all_suppliers(self):
        return list(self.db.supplier.find({}, {'_id': 0}))

    def get_suppliers_by_nation(self, nation_key):
        return list(self.db.supplier.find({"s_nationkey": str(nation_key)}, {'_id': 0}))

    def get_partsupp_for_part(self, partkey):
        return list(self.db.partsupp.find({"ps_partkey": str(partkey)}, {'_id': 0}))

    def get_lowest_cost_supplier_for_part(self, partkey):
        pipeline = [
            {"$match": {"ps_partkey": str(partkey)}},
            {"$addFields": {"ps_supplycost_num": {"$toDouble": "$ps_supplycost"}}},
            {"$sort": {"ps_supplycost_num": 1}},
            {"$limit": 1},
            {"$lookup": {
                "from": "supplier",
                "let": {"suppkey": "$ps_suppkey"},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$s_suppkey", "$$suppkey"]}}},
                    {"$project": {"_id": 0, "s_name": 1, "s_acctbal": 1}}
                ],
                "as": "supplier_info"
            }},
            {"$unwind": "$supplier_info"},
            {"$project": {"_id": 0, "ps_partkey": 1, "ps_suppkey": 1, "ps_supplycost": 1, "s_name": "$supplier_info.s_name", "s_acctbal": "$supplier_info.s_acctbal"}}
        ]
        res = list(self.db.partsupp.aggregate(pipeline))
        return res[0] if res else None

    def count_suppliers_per_part(self):
        pipeline = [
            {"$group": {"_id": "$ps_partkey", "supplier_count": {"$sum": 1}}},
            {"$project": {"_id": 0, "partkey": "$_id", "supplier_count": 1}},
            {"$sort": {"partkey": 1}}
        ]
        return list(self.db.partsupp.aggregate(pipeline))

    def avg_supplycost_by_part_size(self):
        pipeline = [
            {"$lookup": {
                "from": "part",
                "localField": "ps_partkey",
                "foreignField": "p_partkey",
                "as": "part_info"
            }},
            {"$unwind": "$part_info"},
            {"$addFields": {"supply_cost_num": {"$toDouble": "$ps_supplycost"}, "part_size_num": {"$toInt": "$part_info.p_size"}}},
            {"$group": {"_id": "$part_size_num", "avg_supplycost": {"$avg": "$supply_cost_num"}}},
            {"$project": {"_id": 0, "p_size": "$_id", "avg_supplycost": 1}},
            {"$sort": {"p_size": 1}}
        ]
        return list(self.db.partsupp.aggregate(pipeline))
