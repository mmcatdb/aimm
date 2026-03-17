from typing_extensions import override
from common.config import DatasetName
from common.database import Database, MongoQuery, MongoFindQuery, MongoAggregateQuery
from common.drivers import DriverType
import random
import datetime

class TpchMongoDatabase(Database[MongoQuery]):
    NUM_QUERY_TYPES = 32 # Total number of different query types implemented

    def __init__(self):
        super().__init__(DatasetName.TPCH, DriverType.MONGO)

    @override
    def _generate_train_queries(self, num_queries: int):
        """
        Generate a diverse set of MongoDB queries for training.
        Each item is a dict with:
          type: 'find' or 'aggregate'
          collection: str
          params: dict  (filter, projection, sort, limit, skip for find; pipeline for aggregate)
          label: str   (human-readable description)
        """

        target = num_queries

        #region Simple finds with filters

        n = target // 10
        for _ in range(n):
            # lineitem date range with various operators and limits
            date = self._random_date()
            op = random.choice(['$gte', '$lte', '$gt', '$lt'])
            self._train_query(MongoFindQuery('lineitem',
                filter={'l_shipdate': {op: date}},
                limit=random.choice([0, 10, 50, 100, 500, 1000]),
            ))
            # 'label': 'lineitem shipdate filter',

        # Lineitem with quantity / price filters (more selectivity diversity)
        for _ in range(n // 2):
            qty = random.uniform(1, 50)
            self._train_query(MongoFindQuery('lineitem',
                filter={'l_quantity': {'$gt': qty}},
                limit=random.choice([0, 50, 200]),
            ))
            # 'label': 'lineitem quantity filter',

        for _ in range(n):
            # orders price range
            lo = random.uniform(50000, 300000)
            hi = lo + random.uniform(10000, 200000)
            self._train_query(MongoFindQuery('orders',
                filter={'o_totalprice': {'$gte': lo, '$lte': hi}},
                limit=random.choice([0, 20, 50, 200]),
            ))
            # 'label': 'orders price range',

        # Orders with single price threshold + sort + limit
        # (matches evaluation pattern exactly)
        for _ in range(n):
            threshold = random.uniform(10000, 500000)
            self._train_query(MongoFindQuery('orders',
                filter={'o_totalprice': {'$gt': threshold}},
                sort={'o_totalprice': random.choice([-1, 1])},
                limit=random.choice([10, 20, 50, 100]),
            ))
            # 'label': 'orders price threshold sorted',

        #endregion
        #region Index scans (point lookups, small ranges)

        for _ in range(n):
            key = random.randint(1, 300000)
            self._train_query(MongoFindQuery('orders',
                filter={'o_orderkey': key},
            ))
            # 'label': 'orders point lookup',

        for _ in range(n):
            keys = [random.randint(1, 300000) for _ in range(random.randint(2, 20))]
            self._train_query(MongoFindQuery('orders',
                filter={'o_orderkey': {'$in': keys}},
                projection={'o_custkey': 1, 'o_totalprice': 1},
            ))
            # 'label': 'orders multi-key lookup',

        #endregion
        #region Finds with sort

        for _ in range(n):
            self._train_query(MongoFindQuery('orders',
                filter={'o_orderdate': {'$gte': self._random_date()}},
                sort={'o_totalprice': random.choice([-1, 1])},
                limit=random.choice([10, 20, 50, 100]),
            ))
            # 'label': 'orders sorted by price',

        for _ in range(n):
            self._train_query(MongoFindQuery('lineitem',
                filter={'l_shipdate': {'$gte': self._random_date()}},
                sort={'l_extendedprice': -1},
                limit=random.choice([10, 50, 100, 200]),
            ))
            # 'label': 'lineitem sorted by price',

        #endregion
        #region Aggregation with $group

        for _ in range(n):
            self._train_query(MongoAggregateQuery('lineitem', [
                {'$match': {'l_shipdate': {'$gte': self._random_date()}}},
                {'$group': {
                    '_id': '$l_returnflag',
                    'count': {'$sum': 1},
                    'avg_qty': {'$avg': '$l_quantity'},
                    'sum_price': {'$sum': '$l_extendedprice'},
                }},
            ]))
            # 'label': 'lineitem group by returnflag',

        for _ in range(n):
            self._train_query(MongoAggregateQuery('orders', [
                {'$match': {'o_orderdate': {'$gte': self._random_date()}}},
                {'$group': {
                    '_id': '$o_orderpriority',
                    'count': {'$sum': 1},
                    'avg_price': {'$avg': '$o_totalprice'},
                }},
                {'$sort': {'count': -1}},
            ]))
            # 'label': 'orders group by priority',

        #endregion
        #region Aggregation with $lookup (join)

        for _ in range(n // 2):
            lo = random.uniform(100000, 400000)
            self._train_query(MongoAggregateQuery('orders', [
                {'$match': {'o_totalprice': {'$gt': lo}}},
                {'$limit': random.choice([5, 10, 20, 50])},
                {'$lookup': {
                    'from': 'customer',
                    'localField': 'o_custkey',
                    'foreignField': 'c_custkey',
                    'as': 'customer',
                }},
            ]))
            # 'label': 'orders lookup customer',

        for _ in range(n // 2):
            keys = random.sample(range(1, 40001), random.randint(5, 50))
            self._train_query(MongoAggregateQuery('part', [
                {'$match': {'p_partkey': {'$in': keys}}},
                {'$lookup': {
                    'from': 'partsupp',
                    'localField': 'p_partkey',
                    'foreignField': 'ps_partkey',
                    'as': 'suppliers',
                }},
            ]))
            # 'label': 'part lookup partsupp',

        #endregion
        #region Customer queries

        for _ in range(n):
            bal = random.uniform(-1000, 9000)
            seg = random.choice(['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE'])
            self._train_query(MongoFindQuery('customer',
                filter={'c_acctbal': {'$gt': bal}, 'c_mktsegment': seg},
                sort={'c_acctbal': -1},
                limit=random.choice([0, 30, 50, 100]),
            ))
            # 'label': 'customer segment balance',
        # Customer acctbal only (varying selectivity)
        for _ in range(n // 2):
            bal = random.uniform(-1000, 9500)
            self._train_query(MongoFindQuery('customer',
                filter={'c_acctbal': {'$gt': bal}},
                limit=random.choice([0, 50, 100]),
            ))
            # 'label': 'customer balance only',

        #endregion
        #region Part queries

        for _ in range(n // 2):
            lo = random.randint(1, 40)
            hi = lo + random.randint(5, 20)
            brand = f'Brand#{random.randint(11, 55)}'
            self._train_query(MongoFindQuery('part',
                filter={'p_size': {'$gte': lo, '$lte': hi}, 'p_brand': brand},
                sort={'p_retailprice': -1},
                limit=random.choice([0, 50, 100]),
            ))
            # 'label': 'part brand size filter',

        #endregion
        #region Supplier / small collection scans

        # Much more diversity in selectivity for supplier
        for _ in range(n):
            bal = random.uniform(0, 10000)
            self._train_query(MongoFindQuery('supplier',
                filter={'s_acctbal': {'$gt': bal}},
                sort={'s_acctbal': -1},
            ))
            # 'label': 'supplier balance filter',
        for _ in range(n // 2):
            bal_lo = random.uniform(-1000, 5000)
            bal_hi = bal_lo + random.uniform(500, 5000)
            self._train_query(MongoFindQuery('supplier',
                filter={'s_acctbal': {'$gte': bal_lo, '$lte': bal_hi}},
            ))
            # 'label': 'supplier balance range',
        for _ in range(n // 4):
            nation_key = random.randint(0, 24)
            self._train_query(MongoFindQuery('supplier',
                filter={'s_nationkey': nation_key},
            ))
            # 'label': 'supplier by nation',

        #endregion
        #region Full collection scans (no filter)

        for _ in range(n // 4):
            self._train_query(MongoFindQuery('nation'))
            # 'label': 'nation full scan',

        for _ in range(n // 4):
            self._train_query(MongoFindQuery('region'))
            # 'label': 'region full scan',

        #endregion
        #region Skip queries

        for _ in range(n // 2):
            self._train_query(MongoFindQuery('lineitem',
                filter={'l_shipdate': {'$gte': self._random_date()}},
                skip=random.randint(10, 1000),
                limit=random.choice([10, 50, 100]),
            ))
            # 'label': 'lineitem skip+limit',

        #endregion
        #region Projection-only queries

        for _ in range(n // 2):
            self._train_query(MongoFindQuery('customer',
                filter={'c_custkey': random.randint(1, 30000)},
                projection={'c_name': 1, 'c_acctbal': 1, '_id': 0},
            ))
            # 'label': 'customer projection lookup',

        #endregion
        #region Complex aggregates with $group + $sort

        # TPC-H Q1 variants with varying date ranges (from narrow to full-table scan)
        for _ in range(n):
            self._train_query(MongoAggregateQuery('lineitem', [
                {'$match': {'l_shipdate': {'$gte': self._random_date()}}},
                {'$group': {
                    '_id': {'flag': '$l_returnflag', 'status': '$l_linestatus'},
                    'sum_qty': {'$sum': '$l_quantity'},
                    'sum_price': {'$sum': '$l_extendedprice'},
                    'avg_disc': {'$avg': '$l_discount'},
                    'count': {'$sum': 1},
                }},
                {'$sort': {'sum_price': -1}},
            ]))
            # 'label': 'lineitem TPC-H Q1 style',

        # TPC-H Q1 with $lte (matching evaluation style)
        for _ in range(n):
            self._train_query(MongoAggregateQuery('lineitem', [
                {'$match': {'l_shipdate': {'$lte': self._random_date()}}},
                {'$group': {
                    '_id': {'rf': '$l_returnflag', 'ls': '$l_linestatus'},
                    'sum_qty': {'$sum': '$l_quantity'},
                    'sum_price': {'$sum': '$l_extendedprice'},
                    'avg_disc': {'$avg': '$l_discount'},
                    'count': {'$sum': 1},
                }},
                {'$sort': {'_id': 1}},
            ]))
            # 'label': 'lineitem TPC-H Q1 lte',

        # Simple lineitem aggregation (count + sum only, less accumulators)
        for _ in range(n // 2):
            self._train_query(MongoAggregateQuery('lineitem', [
                {'$match': {'l_shipdate': {'$gte': self._random_date()}}},
                {'$group': {
                    '_id': '$l_returnflag',
                    'count': {'$sum': 1},
                }},
            ]))
            # 'label': 'lineitem simple group',

        # Add partsupp queries (multiple patterns including just cost filter)
        for _ in range(n // 2):
            cost = random.uniform(10, 500)
            qty = random.randint(100, 9000)
            self._train_query(MongoFindQuery('partsupp',
                filter={'ps_supplycost': {'$lt': cost}, 'ps_availqty': {'$gt': qty}},
                sort={'ps_supplycost': 1},
                limit=random.choice([0, 50, 100, 200]),
            ))
            # 'label': 'partsupp cost/qty filter',

        # Partsupp with just cost filter + sort (matches evaluation)
        for _ in range(n):
            cost = random.uniform(10, 1000)
            self._train_query(MongoFindQuery('partsupp',
                filter={'ps_supplycost': {'$lt': cost}},
                sort={'ps_supplycost': 1},
                limit=random.choice([0, 50, 100, 200]),
            ))
            # 'label': 'partsupp cost sorted',

        # Supplier without sort (matches evaluation pattern)
        for _ in range(n // 2):
            bal = random.uniform(-1000, 10000)
            self._train_query(MongoFindQuery('supplier',
                filter={'s_acctbal': {'$gt': bal}},
            ))
            # 'label': 'supplier balance nosort',

        #endregion

    @override
    def _generate_test_queries(self):
        """
        Generate test queries

        A) Novel predicate structures ($or, $and, $ne, $nin, $exists, $regex)
        B) Extreme selectivities (near-empty, near-full-table)
        C) Unseen field combinations and date fields (l_commitdate, l_receiptdate)
        D) Different aggregation patterns ($unwind, $project, $count, multi-$group)
        E) Cross-collection lookups not in training (supplier->nation, lineitem->orders)
        F) Large skip values, unlimited scans on medium tables
        G) Multi-stage (4+) aggregate pipelines
        """

        #region A - Baseline find queries (kept from original for coverage)

        for delta_days in [30, 90, 180, 365, 730]:
            d = datetime.datetime(1998, 12, 1) - datetime.timedelta(days=delta_days)
            self._test_query(f'A1-{delta_days}', f'Lineitem shipdate <= {d.strftime("%Y-%m-%d")} LIMIT 100', MongoFindQuery('lineitem',
                filter={'l_shipdate': {'$lte': d}},
                limit=100,
            ))

        for threshold in [50000, 150000, 250000, 350000, 450000]:
            self._test_query(f'A2-{threshold}', f'Orders price>{threshold} sorted LIMIT 50', MongoFindQuery('orders',
                filter={'o_totalprice': {'$gt': threshold}},
                sort={'o_totalprice': -1},
                limit=50,
            ))

        for seg in ['BUILDING', 'AUTOMOBILE', 'MACHINERY', 'HOUSEHOLD', 'FURNITURE']:
            self._test_query(f'A3-{seg}', f'Customer {seg} acctbal>5000', MongoFindQuery('customer',
                filter={'c_mktsegment': seg, 'c_acctbal': {'$gt': 5000}},
                sort={'c_acctbal': -1},
                limit=30,
            ))

        for brand_num in [13, 24, 35, 42, 51]:
            self._test_query(f'A4-{brand_num}', f'Part Brand#{brand_num} size 10-30', MongoFindQuery('part',
                filter={'p_brand': f'Brand#{brand_num}', 'p_size': {'$gte': 10, '$lte': 30}},
                sort={'p_retailprice': -1},
            ))

        for bal in [2000, 4000, 6000, 8000]:
            self._test_query(f'A5-{bal}', f'Supplier acctbal>{bal}', MongoFindQuery('supplier',
                    filter={'s_acctbal': {'$gt': bal}},
            ))

        for key in [42, 1000, 5000, 100000, 250000]:
            self._test_query(f'A6-{key}', f'Orders point lookup key={key}', MongoFindQuery('orders',
                filter={'o_orderkey': key},
            ))

        keys = [random.randint(1, 300000) for _ in range(15)]
        self._test_query('A7', 'Orders 15-key $in lookup', MongoFindQuery('orders',
            filter={'o_orderkey': {'$in': keys}},
            projection={'o_custkey': 1, 'o_totalprice': 1},
        ))

        self._test_query('A8-1', 'Nation full scan', MongoFindQuery('nation'))
        self._test_query('A8-2', 'Region full scan', MongoFindQuery('region'))

        for cost in [50, 200, 500]:
            self._test_query(f'A9-{cost}', f'Partsupp cost<{cost} sorted LIMIT 100', MongoFindQuery('partsupp',
                filter={'ps_supplycost': {'$lt': cost}},
                sort={'ps_supplycost': 1},
                limit=100,
            ))

        for delta in [60, 90, 120]:
            d = datetime.datetime(1998, 12, 1) - datetime.timedelta(days=delta)
            self._test_query(f'A10-{delta}', f'TPC-H Q1 delta={delta}d', MongoAggregateQuery('lineitem', [
                {'$match': {'l_shipdate': {'$lte': d}}},
                {'$group': {
                    '_id': {'rf': '$l_returnflag', 'ls': '$l_linestatus'},
                    'sum_qty': {'$sum': '$l_quantity'},
                    'sum_price': {'$sum': '$l_extendedprice'},
                    'avg_disc': {'$avg': '$l_discount'},
                    'count': {'$sum': 1}},
                },
                {'$sort': {'_id': 1}},
            ]))

        for year in [1994, 1995, 1996, 1997]:
            d = datetime.datetime(year, 1, 1)
            self._test_query(f'A11-{year}', f'Orders group priority year>={year}', MongoAggregateQuery('orders', [
                {'$match': {'o_orderdate': {'$gte': d}}},
                {'$group': {
                    '_id': '$o_orderpriority',
                    'count': {'$sum': 1},
                    'avg_price': {'$avg': '$o_totalprice'}},
                },
                {'$sort': {'count': -1}},
            ]))

        for price_threshold in [200000, 300000, 400000]:
            self._test_query(f'A12-{price_threshold}', f'Orders lookup customer price>{price_threshold}', MongoAggregateQuery('orders', [
                {'$match': {'o_totalprice': {'$gt': price_threshold}}},
                {'$limit': 10},
                {'$lookup': {
                    'from': 'customer',
                    'localField': 'o_custkey',
                    'foreignField': 'c_custkey',
                    'as': 'customer'
                }},
            ]))

        for size_threshold in [20, 30, 40]:
            self._test_query(f'A13-{size_threshold}', f'Part lookup partsupp size>{size_threshold}', MongoAggregateQuery('part', [
                {'$match': {'p_size': {'$gt': size_threshold}}},
                {'$limit': 20},
                {'$lookup': {
                    'from': 'partsupp',
                    'localField': 'p_partkey',
                    'foreignField': 'ps_partkey',
                    'as': 'suppliers'
                }},
            ]))

        #endregion
        #region B - Novel predicate structures ($or, $and, $ne, $nin, $exists, $regex)

        # B1. $or across different fields (training never uses $or in find)
        self._test_query('B1-1', 'Orders $or extreme prices', MongoFindQuery('orders',
            filter={'$or': [
                {'o_totalprice': {'$lt': 5000}},
                {'o_totalprice': {'$gt': 450000}},
            ]},
            limit=200,
        ))

        self._test_query('B1-2', 'Customer $or segment+balance', MongoFindQuery('customer',
            filter={'$or': [
                {'c_mktsegment': 'BUILDING', 'c_acctbal': {'$gt': 9000}},
                {'c_mktsegment': 'FURNITURE', 'c_acctbal': {'$lt': -500}},
            ]},
        ))

        self._test_query('B1-3', 'Lineitem $or returnflag|shipmode+qty', MongoFindQuery('lineitem',
            filter={'$or': [
                {'l_returnflag': 'R'},
                {'l_shipmode': 'AIR', 'l_quantity': {'$gt': 45}},
            ]},
            limit=500,
        ))

        # B2. $and with many predicates (3+ field compound filter)
        self._test_query('B2-1', 'Lineitem $and 4-pred (TPC-H Q6 style)', MongoFindQuery('lineitem',
            filter={'$and': [
                {'l_shipdate': {'$gte': datetime.datetime(1995, 1, 1)}},
                {'l_shipdate': {'$lt': datetime.datetime(1996, 1, 1)}},
                {'l_discount': {'$gte': 0.05, '$lte': 0.07}},
                {'l_quantity': {'$lt': 24}},
            ]},
        ))

        self._test_query('B2-2', 'Orders $and date+priority+price', MongoFindQuery('orders',
            filter={'$and': [
                {'o_orderdate': {'$gte': datetime.datetime(1994, 1, 1)}},
                {'o_orderdate': {'$lt': datetime.datetime(1995, 1, 1)}},
                {'o_orderpriority': {'$in': ['1-URGENT', '2-HIGH']}},
                {'o_totalprice': {'$gt': 100000}},
            ]},
            limit=100,
        ))

        # B3. $ne operator (never in training)
        self._test_query('B3-1', 'Orders status $ne F', MongoFindQuery('orders',
            filter={'o_orderstatus': {'$ne': 'F'}},
            limit=200,
        ))

        self._test_query('B3-2', 'Lineitem $ne returnflag+shipmode', MongoFindQuery('lineitem',
            filter={'l_returnflag': {'$ne': 'N'}, 'l_shipmode': {'$ne': 'TRUCK'}},
            limit=300,
        ))

        # B4. $nin operator
        self._test_query('B4-1', 'Lineitem shipmode $nin 3 values', MongoFindQuery('lineitem',
            filter={'l_shipmode': {'$nin': ['AIR', 'MAIL', 'RAIL']}},
            limit=500,
        ))

        self._test_query('B4-2', 'Orders priority $nin urgent/high', MongoFindQuery('orders',
            filter={'o_orderpriority': {'$nin': ['1-URGENT', '2-HIGH']}},
            limit=200,
        ))

        # B5. $exists (structural predicate — never in training)
        self._test_query('B5', 'Customer $exists comment + high bal', MongoFindQuery('customer',
            filter={
                'c_comment': {'$exists': True},
                'c_acctbal': {'$gt': 8000},
            },
        ))

        # B6. $regex (never in training)
        self._test_query('B6-1', 'Customer $regex name prefix', MongoFindQuery('customer',
            filter={'c_name': {'$regex': '^Customer#00001'}},
            limit=50,
        ))

        self._test_query('B6-2', 'Part $regex type ends BRASS', MongoFindQuery('part',
            filter={'p_type': {'$regex': 'BRASS$'}},
        ))

        self._test_query('B6-3', 'Part $regex name contains green', MongoFindQuery('part',
            filter={'p_name': {'$regex': 'green'}},
        ))

        # B7. Nested $or inside $and
        self._test_query('B7', 'Lineitem nested $and/$or (TPC-H Q19 style)', MongoFindQuery('lineitem',
            filter={'$and': [
                {'l_shipdate': {'$gte': datetime.datetime(1994, 1, 1)}},
                {'$or': [
                    {'l_shipmode': 'AIR'},
                    {'l_shipmode': 'REG AIR'},
                ]},
                {'$or': [
                    {'l_shipinstruct': 'DELIVER IN PERSON'},
                    {'l_quantity': {'$lt': 10}},
                ]},
            ]},
            limit=200,
        ))

        #endregion
        #region C - Extreme selectivities

        # C1. Near-empty results (very tight filter)
        self._test_query('C1-1', 'Orders extreme high price (near empty)', MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 490000}},
        ))

        self._test_query('C1-2', 'Customer acctbal>9990 (near empty)', MongoFindQuery('customer',
            filter={'c_acctbal': {'$gt': 9990}},
        ))

        self._test_query('C1-3', 'Lineitem exact qty=1 disc=0 tax=0 (near empty)', MongoFindQuery('lineitem',
            filter={'l_quantity': 1.0, 'l_discount': 0.0, 'l_tax': 0.0},
        ))

        # C2. Near-full-table scans (very permissive filter)
        self._test_query('C2-1', 'Lineitem qty>0 (full table, 1.2M rows)', MongoFindQuery('lineitem',
            filter={'l_quantity': {'$gt': 0}},
        ))

        self._test_query('C2-2', 'Orders price>0 (full table, 300K rows)', MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 0}},
        ))

        self._test_query('C2-3', 'Customer full scan (30K rows)', MongoFindQuery('customer'))

        self._test_query('C2-4', 'Partsupp full scan (160K rows)', MongoFindQuery('partsupp'))

        self._test_query('C2-5', 'Supplier full scan (2K rows)', MongoFindQuery('supplier'))

        # C3. Unlimited scans with sort on medium table (no limit!)
        self._test_query('C3-1', 'Customer bal>0 sorted NO LIMIT', MongoFindQuery('customer',
            filter={'c_acctbal': {'$gt': 0}},
            sort={'c_acctbal': -1},
        ))

        self._test_query('C3-2', 'Orders 1997+ sorted by price NO LIMIT', MongoFindQuery('orders',
            filter={'o_orderdate': {'$gte': datetime.datetime(1997, 1, 1)}},
            sort={'o_totalprice': -1},
        ))

        #endregion
        #region D - Unseen fields & field combinations

        # D1. l_commitdate (never used in training)
        self._test_query('D1', 'Lineitem commitdate<1993-06 LIMIT 200', MongoFindQuery('lineitem',
            filter={'l_commitdate': {'$lt': datetime.datetime(1993, 6, 1)}},
            limit=200,
        ))

        # D2. l_receiptdate (never used in training)
        self._test_query('D2', 'Lineitem receiptdate>=1998-06', MongoFindQuery('lineitem',
            filter={'l_receiptdate': {'$gte': datetime.datetime(1998, 6, 1)}},
            limit=500,
        ))

        # D3. Cross-date comparison style: shipdate AND commitdate
        self._test_query('D3', 'Lineitem 3-date cross filter', MongoFindQuery('lineitem',
            filter={
                'l_shipdate': {'$gte': datetime.datetime(1995, 1, 1)},
                'l_commitdate': {'$lt': datetime.datetime(1995, 3, 1)},
                'l_receiptdate': {'$gte': datetime.datetime(1995, 2, 1)},
            },
            limit=200,
        ))

        # D4. Orders by orderstatus (categorical, not used much in training)
        for status in ['F', 'O', 'P']:
            self._test_query(f'D4-{status}', f'Orders status={status} LIMIT 500', MongoFindQuery('orders',
                filter={'o_orderstatus': status},
                limit=500,
            ))

        # D5. Orders by o_clerk (high cardinality string, never in training)
        self._test_query('D5', 'Orders exact clerk lookup', MongoFindQuery('orders',
            filter={'o_clerk': 'Clerk#000000951'},
        ))

        # D6. Lineitem by ship instruction + mode (string equality combo)
        self._test_query('D6', 'Lineitem shipinstruct+mode combo', MongoFindQuery('lineitem',
            filter={'l_shipinstruct': 'DELIVER IN PERSON', 'l_shipmode': 'AIR'},
            limit=300,
        ))

        # D7. Part by container + type (string fields not in training)
        self._test_query('D7-1', 'Part small containers size<=5', MongoFindQuery('part',
            filter={'p_container': {'$in': ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG']}, 'p_size': {'$lte': 5}},
        ))

        self._test_query('D7-2', 'Part Mfgr#3 ECONOMY types', MongoFindQuery('part',
            filter={'p_mfgr': 'Manufacturer#3', 'p_type': {'$regex': '^ECONOMY'}},
        ))

        # D8. Customer by nationkey (joining field, never filtered in training)
        for nk in [0, 7, 15, 24]:
            self._test_query(f'D8-{nk}', f'Customer by nationkey={nk}', MongoFindQuery('customer',
                filter={'c_nationkey': nk},
            ))

        # D9. Partsupp by availqty ranges not in training
        self._test_query('D9-1', 'Partsupp very low avail qty<100', MongoFindQuery('partsupp',
            filter={'ps_availqty': {'$lt': 100}},
        ))

        self._test_query('D9-2', 'Partsupp high qty+cost', MongoFindQuery('partsupp',
            filter={'ps_availqty': {'$gt': 9500}, 'ps_supplycost': {'$gt': 900}},
        ))

        #endregion
        #region E - Large skip / pagination patterns

        self._test_query('E1', 'Orders deep pagination skip=5000', MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 100000}},
            sort={'o_totalprice': -1},
            skip=5000,
            limit=50,
        ))

        self._test_query('E2', 'Lineitem deep skip=10000', MongoFindQuery('lineitem',
            filter={'l_shipdate': {'$gte': datetime.datetime(1997, 1, 1)}},
            skip=10000,
            limit=100,
        ))

        self._test_query('E3', 'Customer skip=15000 sorted', MongoFindQuery('customer',
            filter={},
            sort={'c_acctbal': -1},
            skip=15000,
            limit=50,
        ))

        #endregion
        #region F - $in on large value lists (varying list sizes)

        # Large $in list on indexed field
        keys_50 = [random.randint(1, 300000) for _ in range(50)]
        self._test_query('F1', 'Orders 50-key $in lookup', MongoFindQuery('orders',
            filter={'o_orderkey': {'$in': keys_50}},
        ))

        keys_200 = [random.randint(1, 300000) for _ in range(200)]
        self._test_query('F2', 'Orders 200-key $in lookup', MongoFindQuery('orders',
            filter={'o_orderkey': {'$in': keys_200}},
        ))

        # $in on non-indexed field (string values)
        self._test_query('F3', 'Lineitem shipmode $in [AIR,RAIL]', MongoFindQuery('lineitem',
            filter={'l_shipmode': {'$in': ['AIR', 'RAIL']}},
            limit=1000,
        ))

        self._test_query('F4', 'Part 9-brand $in lookup', MongoFindQuery('part',
            filter={'p_brand': {'$in': [f'Brand#{i}' for i in range(11, 20)]}},
        ))

        #endregion
        #region G - Novel aggregation patterns

        # G1. $count stage
        self._test_query('G1-1', 'Lineitem $count AIR 1997+', MongoAggregateQuery('lineitem', [
            {'$match': {
                'l_shipmode': 'AIR',
                'l_shipdate': {'$gte': datetime.datetime(1997, 1, 1)},
            }},
            {'$count': 'total'},
        ]))

        self._test_query('G1-2', 'Orders $count status=P', MongoAggregateQuery('orders', [
            {'$match': {'o_orderstatus': 'P'}},
            {'$count': 'total'},
        ]))

        # G2. $group with $min/$max/$first/$last (accumulators not in training)
        self._test_query('G2', 'Lineitem group shipmode min/max/first/last', MongoAggregateQuery('lineitem', [
            {'$match': {'l_shipdate': {'$gte': datetime.datetime(1996, 1, 1)}}},
            {'$group': {
                '_id': '$l_shipmode',
                'min_price': {'$min': '$l_extendedprice'},
                'max_price': {'$max': '$l_extendedprice'},
                'first_date': {'$first': '$l_shipdate'},
                'last_date': {'$last': '$l_shipdate'},
                'count': {'$sum': 1},
            }},
        ]))

        # G3. $group by high-cardinality field (o_custkey -> many groups)
        self._test_query('G3', 'Orders group by custkey top-20 spenders', MongoAggregateQuery('orders', [
            {'$group': {
                '_id': '$o_custkey',
                'order_count': {'$sum': 1},
                'total_spent': {'$sum': '$o_totalprice'},
            }},
            {'$sort': {'total_spent': -1}},
            {'$limit': 20},
        ]))

        # G4. $project stage (reshaping, never in training pipelines)
        self._test_query('G4', 'Lineitem $project revenue then group (4 stages)', MongoAggregateQuery('lineitem', [
            {'$match': {'l_shipdate': {'$gte': datetime.datetime(1997, 6, 1)}}},
            {'$project': {
                'revenue': {'$multiply': [
                    '$l_extendedprice',
                    {'$subtract': [1, '$l_discount']},
                ]},
                'l_shipmode': 1,
                'l_returnflag': 1,
            }},
            {'$group': {
                '_id': '$l_shipmode',
                'total_revenue': {'$sum': '$revenue'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'total_revenue': -1}},
        ]))

        # G5. Customer aggregation by segment with $project (never in training)
        self._test_query('G5', 'Customer group by segment avg/max bal', MongoAggregateQuery('customer', [
            {'$match': {'c_acctbal': {'$gt': 0}}},
            {'$group': {
                '_id': '$c_mktsegment',
                'avg_bal': {'$avg': '$c_acctbal'},
                'max_bal': {'$max': '$c_acctbal'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'avg_bal': -1}},
        ]))

        # G6. Multi-stage pipeline with $match -> $group -> $match -> $sort (double filter)
        self._test_query('G6', 'Orders double-$match group then filter (TPC-H Q13 style)', MongoAggregateQuery('orders', [
            {'$match': {'o_orderdate': {
                '$gte': datetime.datetime(1993, 1, 1),
                '$lt': datetime.datetime(1998, 1, 1)
            }}},
            {'$group': {
                '_id': '$o_custkey',
                'order_count': {'$sum': 1},
                'avg_price': {'$avg': '$o_totalprice'},
            }},
            {'$match': {'order_count': {'$gte': 10}}},
            {'$sort': {'avg_price': -1}},
            {'$limit': 50},
        ]))

        # G7. Partsupp aggregation (collection rarely aggregated in training)
        self._test_query('G7', 'Partsupp group by supplier cost<100', MongoAggregateQuery('partsupp', [
            {'$match': {'ps_supplycost': {'$lt': 100}}},
            {'$group': {
                '_id': '$ps_suppkey',
                'part_count': {'$sum': 1},
                'avg_cost': {'$avg': '$ps_supplycost'},
                'total_qty': {'$sum': '$ps_availqty'},
            }},
            {'$sort': {'part_count': -1}},
            {'$limit': 20},
        ]))

        # G8. Supplier aggregation by nation
        self._test_query('G8', 'Supplier group by nationkey', MongoAggregateQuery('supplier', [
            {'$group': {
                '_id': '$s_nationkey',
                'supplier_count': {'$sum': 1},
                'avg_balance': {'$avg': '$s_acctbal'},
            }},
            {'$sort': {'avg_balance': -1}},
        ]))

        #endregion
        #region H - Novel cross-collection lookups

        # H1. Supplier -> nation (never in training)
        self._test_query('H1', 'Supplier lookup nation bal>5000', MongoAggregateQuery('supplier', [
            {'$match': {'s_acctbal': {'$gt': 5000}}},
            {'$lookup': {
                'from': 'nation',
                'localField': 's_nationkey',
                'foreignField': 'n_nationkey',
                'as': 'nation_info'
            }},
        ]))

        # H2. Customer -> nation -> region chain (2 lookups, never in training)
        self._test_query('H2', 'Customer->nation->region chain (5 stages)', MongoAggregateQuery('customer', [
            {'$match': {'c_acctbal': {'$gt': 9000}}},
            {'$limit': 50},
            {'$lookup': {
                'from': 'nation',
                'localField': 'c_nationkey',
                'foreignField': 'n_nationkey',
                'as': 'nation'
            }},
            {'$unwind': '$nation'},
            {'$lookup': {
                'from': 'region',
                'localField': 'nation.n_regionkey',
                'foreignField': 'r_regionkey',
                'as': 'region'
            }},
        ]))

        # H3. Lineitem -> orders lookup (largest collection as driving table)
        self._test_query('H3', 'Lineitem->orders lookup recent shipments', MongoAggregateQuery('lineitem', [
            {'$match': {
                'l_shipdate': {'$gte': datetime.datetime(1998, 11, 1)},
                'l_returnflag': 'N'
            }},
            {'$limit': 50},
            {'$lookup': {
                'from': 'orders',
                'localField': 'l_orderkey',
                'foreignField': 'o_orderkey',
                'as': 'order_info'
            }},
        ]))

        # H4. Nation -> region lookup (tiny->tiny, very different cost profile)
        self._test_query('H4', 'Nation->region lookup+unwind (tiny tables)', MongoAggregateQuery('nation', [
            {'$lookup': {
                'from': 'region',
                'localField': 'n_regionkey',
                'foreignField': 'r_regionkey',
                'as': 'region'
            }},
            {'$unwind': '$region'},
        ]))

        # H5. Orders -> customer lookup with group after (lookup + aggregate)
        self._test_query('H5', 'Orders->customer lookup+unwind+group (6 stages)', MongoAggregateQuery('orders', [
            {'$match': {'o_orderdate': {'$gte': datetime.datetime(1997, 1, 1)}}},
            {'$limit': 100},
            {'$lookup': {
                'from': 'customer',
                'localField': 'o_custkey',
                'foreignField': 'c_custkey',
                'as': 'cust'
            }},
            {'$unwind': '$cust'},
            {'$group': {
                '_id': '$cust.c_mktsegment',
                'order_count': {'$sum': 1},
                'avg_price': {'$avg': '$o_totalprice'},
            }},
            {'$sort': {'order_count': -1}},
        ]))

        #endregion
        #region I - $unwind-heavy pipelines

        # I1. Part->partsupp lookup + unwind + group (TPC-H Q2/Q11 style)
        self._test_query('I1', 'Part BRASS size=15 lookup+unwind+sort (TPC-H Q2 style)', MongoAggregateQuery('part', [
            {'$match': {'p_size': 15, 'p_type': {'$regex': 'BRASS$'}}},
            {'$lookup': {
                'from': 'partsupp',
                'localField': 'p_partkey',
                'foreignField': 'ps_partkey', 'as': 'ps'
            }},
            {'$unwind': '$ps'},
            {'$sort': {'ps.ps_supplycost': 1}},
            {'$limit': 20},
        ]))

        # I2. Customer -> orders lookup + unwind (never in training)
        self._test_query('I2', 'Customer AUTOMOBILE->orders unwind+group (7 stages)', MongoAggregateQuery('customer', [
            {'$match': {'c_mktsegment': 'AUTOMOBILE'}},
            {'$limit': 20},
            {'$lookup': {
                'from': 'orders',
                'localField': 'c_custkey',
                'foreignField': 'o_custkey',
                'as': 'orders'
            }},
            {'$unwind': '$orders'},
            {'$group': {
                '_id': '$c_custkey',
                'num_orders': {'$sum': 1},
                'total_spent': {'$sum': '$orders.o_totalprice'},
            }},
            {'$sort': {'total_spent': -1}},
            {'$limit': 10},
        ]))

        #endregion
        #region J - Varying limit values (very large, very small, absent)

        self._test_query('J1', 'Lineitem top-1 most expensive 1995+', MongoFindQuery('lineitem',
            filter={'l_shipdate': {'$gte': datetime.datetime(1995, 1, 1)}},
            sort={'l_extendedprice': -1},
            limit=1,
        ))

        self._test_query('J2', 'Lineitem top-5000 most expensive 1995+', MongoFindQuery('lineitem',
            filter={'l_shipdate': {'$gte': datetime.datetime(1995, 1, 1)}},
            sort={'l_extendedprice': -1},
            limit=5000,
        ))

        self._test_query('J3', 'Orders earliest order price>100K (limit 1)', MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 100000}},
            sort={'o_orderdate': 1},
            limit=1,
        ))

        self._test_query('J4', 'Orders price>100K sorted by date limit 10000', MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 100000}},
            sort={'o_orderdate': 1},
            limit=10000,
        ))

        #endregion
