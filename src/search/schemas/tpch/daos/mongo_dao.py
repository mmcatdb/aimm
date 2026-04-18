from typing_extensions import override
from core.drivers import MongoDriver
from .tpch_dao import TpchDAO

class TpchMongoDAO(TpchDAO):
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
    def get_all_lineitems(self):
        return list(self._db.lineitem.find({}, {'_id': 0}))

    @override
    def get_orders_by_daterange(self, start_date, end_date):
        query = {
            'o_orderdate': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        return list(self._db.orders.find(query, {'_id': 0}))

    @override
    def get_all_customers(self):
        return list(self._db.customer.find({}, {'_id': 0}))

    @override
    def get_orders_by_keyrange(self, start_key, end_key):
        query = {
            'o_orderkey': {
                '$gte': start_key,
                '$lte': end_key
            }
        }
        return list(self._db.orders.find(query, {'_id': 0}))

    @override
    def count_orders_by_month(self):
        pipeline = [
            {
                '$group': {
                    '_id': { '$substr': ['$o_orderdate', 0, 7] },
                    'order_count': { '$sum': 1 }
                }
            },
            {
                '$project': {
                    'order_month': '$_id',
                    'order_count': 1,
                    '_id': 0
                }
            },
            { '$sort': { 'order_month': 1 } }
        ]

        return list(self._db.orders.aggregate(pipeline))

    @override
    def get_max_price_by_ship_month(self):
        pipeline = [
            {
                '$group': {
                    '_id': { '$substr': ['$l_shipdate', 0, 7] },
                    'max_price': { '$max': '$l_extendedprice' }
                }
            },
            {
                '$project': {
                    'ship_month': '$_id',
                    'max_price': 1,
                    '_id': 0
                }
            },
            { '$sort': { 'ship_month': 1 } }
        ]
        return list(self._db.lineitem.aggregate(pipeline))

    # --- Part / Supplier / PartSupp ---
    @override
    def get_all_parts(self):
        return list(self._db.part.find({}, {'_id': 0}))

    @override
    def get_parts_by_size_range(self, min_size, max_size):
        query = {'p_size': {'$gte': str(min_size), '$lte': str(max_size)}}
        return list(self._db.part.find(query, {'_id': 0}))

    @override
    def get_all_suppliers(self):
        return list(self._db.supplier.find({}, {'_id': 0}))

    @override
    def get_suppliers_by_nation(self, nation_key):
        return list(self._db.supplier.find({'s_nationkey': str(nation_key)}, {'_id': 0}))

    @override
    def get_partsupp_for_part(self, partkey):
        return list(self._db.partsupp.find({'ps_partkey': str(partkey)}, {'_id': 0}))

    @override
    def get_lowest_cost_supplier_for_part(self, partkey):
        pipeline = [
            {'$match': {'ps_partkey': str(partkey)}},
            {'$addFields': {'ps_supplycost_num': {'$toDouble': '$ps_supplycost'}}},
            {'$sort': {'ps_supplycost_num': 1}},
            {'$limit': 1},
            {'$lookup': {
                'from': 'supplier',
                'let': {'suppkey': '$ps_suppkey'},
                'pipeline': [
                    {'$match': {'$expr': {'$eq': ['$s_suppkey', '$$suppkey']}}},
                    {'$project': {'_id': 0, 's_name': 1, 's_acctbal': 1}}
                ],
                'as': 'supplier_info'
            }},
            {'$unwind': '$supplier_info'},
            {'$project': {'_id': 0, 'ps_partkey': 1, 'ps_suppkey': 1, 'ps_supplycost': 1, 's_name': '$supplier_info.s_name', 's_acctbal': '$supplier_info.s_acctbal'}}
        ]
        res = list(self._db.partsupp.aggregate(pipeline))
        return res[0] if res else None

    @override
    def count_suppliers_per_part(self):
        pipeline = [
            {'$group': {'_id': '$ps_partkey', 'supplier_count': {'$sum': 1}}},
            {'$project': {'_id': 0, 'partkey': '$_id', 'supplier_count': 1}},
            {'$sort': {'partkey': 1}}
        ]
        return list(self._db.partsupp.aggregate(pipeline))

    @override
    def avg_supplycost_by_part_size(self):
        pipeline = [
            {'$lookup': {
                'from': 'part',
                'localField': 'ps_partkey',
                'foreignField': 'p_partkey',
                'as': 'part_info'
            }},
            {'$unwind': '$part_info'},
            {'$addFields': {'supply_cost_num': {'$toDouble': '$ps_supplycost'}, 'part_size_num': {'$toInt': '$part_info.p_size'}}},
            {'$group': {'_id': '$part_size_num', 'avg_supplycost': {'$avg': '$supply_cost_num'}}},
            {'$project': {'_id': 0, 'p_size': '$_id', 'avg_supplycost': 1}},
            {'$sort': {'p_size': 1}}
        ]
        return list(self._db.partsupp.aggregate(pipeline))
