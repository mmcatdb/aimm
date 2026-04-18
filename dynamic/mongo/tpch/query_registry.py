from datetime import datetime, timedelta
from core.drivers import DriverType
from core.query import query, MongoQuery, MongoFindQuery, MongoAggregateQuery
from ...common.tpch.query_registry import TpchQueryRegistry

def export():
    return MongoTpchQueryRegistry()

class MongoTpchQueryRegistry(TpchQueryRegistry[MongoQuery]):

    def __init__(self):
        super().__init__(DriverType.MONGO)

    #region Basic

    @query('basic-0', 'lineitem date range with various operators and limits')
    def _lineitem_date_range(self):
        operator = self._param_choice('operator', ['$gte', '$lte', '$gt', '$lt'])

        return MongoFindQuery('lineitem',
            filter={'l_shipdate': {operator: self._param_date()}},
            limit=self._param_limit(max_order=10)
        )

    @query('basic-1', 'lineitem quantity filter (more selectivity diversity)')
    def _lineitem_quantity(self):
        return MongoFindQuery('lineitem',
            filter={'l_quantity': {'$gt': self._param_int('quantity', 1, 50)}},
            limit=self._param_limit()
        )

    @query('basic-2', 'orders price range')
    def _orders_price_range(self):
        base = self._rng_int(50000, 300000)
        low = self._param('low', lambda: base)
        high = self._param('high', lambda: base + self._rng_int(10000, 200000))
        return MongoFindQuery('orders',
            filter={'o_totalprice': {'$gte': low, '$lte': high}},
            limit=self._param_limit(),
        )

    @query('basic-3', 'orders price threshold sorted')
    def _orders_price_sorted_limit(self):
        return MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': self._param_int('threshold', 10000, 500000)}},
            sort={'o_totalprice': self._param_choice('sort', [1, -1])},
            limit=self._param_limit(),
        )

    @query('basic-4', 'Lineitem shipdate filter')
    def _lineitem_shipdate(self):
        return MongoFindQuery('lineitem',
            filter={'l_shipdate': {'$lte': self._param_date(1996, 1998)}},
            limit=100,
        )

    @query('basic-5', 'Orders price filter + sort')
    def _orders_price_sort(self):
        return MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': self._param_int('threshold', 50000, 450000)}},
            sort={'o_totalprice': -1},
            limit=50,
        )

    @query('basic-6', 'Customer segment balance filter')
    def _customer_segment_balance_sort(self):
        return MongoFindQuery('customer',
            filter={
                'c_mktsegment': self._param_segment(),
                'c_acctbal': {'$gt': 5000}
            },
            sort={'c_acctbal': -1},
            limit=30,
        )

    @query('basic-7', 'Part brand size filter')
    def _part_brand_size(self):
        return MongoFindQuery('part',
            filter={
                'p_brand': self._param_brand(),
                'p_size': {'$gte': 10, '$lte': 30}
            },
            sort={'p_retailprice': -1},
        )

    @query('basic-8', 'Supplier balance filter')
    def _supplier_balance(self):
        return MongoFindQuery('supplier',
            filter={'s_acctbal': {'$gt': self._param_int('balance', 2000, 8000)}},
        )

    @query('basic-9', 'Partsupp cost filter + sort')
    def _partsupp_cost_sort(self):
        return MongoFindQuery('partsupp',
            filter={'ps_supplycost': {'$lt': self._param_float('cost', 50, 500)}},
            sort={'ps_supplycost': 1},
            limit=100,
        )

    #endregion
    #region Index scan
    # (point lookups, small ranges)

    @query('index-scan-0', 'orders point lookup')
    def _orders_point_lookup(self):
        return MongoFindQuery('orders',
            filter={'o_orderkey': self._param_orderkey()},
        )

    @query('index-scan-1', 'orders multi-key lookup')
    def _orders_multi_key_lookup(self):
        return MongoFindQuery('orders',
            filter={'o_orderkey': {'$in': self._param_orderkeys(2, 10)}},
            projection={'o_custkey': 1, 'o_totalprice': 1},
        )

    #endregion
    #region Finds + sort

    @query('sort-0', 'orders sorted by price')
    def _orders_by_price(self):
        return MongoFindQuery('orders',
            filter={'o_orderdate': {'$gte': self._param_date()}},
            sort={'o_totalprice': self._param_choice('sort', [1, -1])},
            limit=self._param_limit(),
        )

    @query('sort-1', 'lineitem sorted by price')
    def _lineitem_by_price(self):
        return MongoFindQuery('lineitem',
            filter={'l_shipdate': {'$gte': self._param_date()}},
            sort={'l_extendedprice': -1},
            limit=self._param_limit(),
        )

    #endregion
    #region $group

    @query('group-0', 'lineitem group by returnflag')
    def _lineitem_group(self):
        return MongoAggregateQuery('lineitem', [
            {'$match': {'l_shipdate': {'$gte': self._param_date()}}},
            {'$group': {
                '_id': '$l_returnflag',
                'count': {'$sum': 1},
                'avg_qty': {'$avg': '$l_quantity'},
                'sum_price': {'$sum': '$l_extendedprice'},
            }},
        ])

    @query('group-1', 'orders group by priority')
    def _orders_group(self):
        return MongoAggregateQuery('orders', [
            {'$match': {'o_orderdate': {'$gte': self._param_date()}}},
            {'$group': {
                '_id': '$o_orderpriority',
                'count': {'$sum': 1},
                'avg_price': {'$avg': '$o_totalprice'},
            }},
            {'$sort': {'count': -1}},
        ])

    @query('group-2', 'TPC-H Q1 date filter')
    def _tpch_date(self):
        return MongoAggregateQuery('lineitem', [
            {'$match': {'l_shipdate': {'$lte': self._param_date(1997, 1998)}}},
            {'$group': {
                '_id': {'rf': '$l_returnflag', 'ls': '$l_linestatus'},
                'sum_qty': {'$sum': '$l_quantity'},
                'sum_price': {'$sum': '$l_extendedprice'},
                'avg_disc': {'$avg': '$l_discount'},
                'count': {'$sum': 1}},
            },
            {'$sort': {'_id': 1}},
        ])

    @query('group-3', 'Orders group priority year filter')
    def _orders_group_priority_year(self):
        return MongoAggregateQuery('orders', [
            {'$match': {'o_orderdate': {'$gte': self._param_date(1994, 1997)}}},
            {'$group': {
                '_id': '$o_orderpriority',
                'count': {'$sum': 1},
                'avg_price': {'$avg': '$o_totalprice'}},
            },
            {'$sort': {'count': -1}},
        ])

    @query('group-4', 'Lineitem group shipmode min/max/first/last')
    def _lineitem_group_shipmode(self):
        return MongoAggregateQuery('lineitem', [
            {'$match': {'l_shipdate': {'$gte': self._param_date(1996, 1996)}}},
            {'$group': {
                '_id': '$l_shipmode',
                'min_price': {'$min': '$l_extendedprice'},
                'max_price': {'$max': '$l_extendedprice'},
                'first_date': {'$first': '$l_shipdate'},
                'last_date': {'$last': '$l_shipdate'},
                'count': {'$sum': 1},
            }},
        ])

    @query('group-5', 'Orders group by custkey top-20 spenders (o_custkey -> many groups)')
    def _orders_group_custkey(self):
        return MongoAggregateQuery('orders', [
            {'$group': {
                '_id': '$o_custkey',
                'order_count': {'$sum': 1},
                'total_spent': {'$sum': '$o_totalprice'},
            }},
            {'$sort': {'total_spent': -1}},
            {'$limit': 20},
        ])

    @query('group-6', 'Lineitem $project revenue then group (4 stages)')
    def _lineitem_project_revenue(self):
        return MongoAggregateQuery('lineitem', [
            {'$match': {'l_shipdate': {'$gte': self._param_date(1997, 1997)}}},
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
        ])

    @query('group-7', 'Customer group by segment avg/max balance')
    def _customer_group_segment(self):
        return MongoAggregateQuery('customer', [
            {'$match': {'c_acctbal': {'$gt': 0}}},
            {'$group': {
                '_id': '$c_mktsegment',
                'avg_bal': {'$avg': '$c_acctbal'},
                'max_bal': {'$max': '$c_acctbal'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'avg_bal': -1}},
        ])

    @query('group-8', 'Orders double-$match group then filter (TPC-H Q13 style)')
    def _orders_double_match_group(self):
        return MongoAggregateQuery('orders', [
            {'$match': {'o_orderdate': {
                '$gte': datetime(1993, 1, 1),
                '$lt': datetime(1998, 1, 1)
            }}},
            {'$group': {
                '_id': '$o_custkey',
                'order_count': {'$sum': 1},
                'avg_price': {'$avg': '$o_totalprice'},
            }},
            {'$match': {'order_count': {'$gte': 10}}},
            {'$sort': {'avg_price': -1}},
            {'$limit': 50},
        ])

    @query('group-9', 'Partsupp group by supplier cost < 100')
    def _partsupp_group_supplier_cost(self):
        return MongoAggregateQuery('partsupp', [
            {'$match': {'ps_supplycost': {'$lt': 100}}},
            {'$group': {
                '_id': '$ps_suppkey',
                'part_count': {'$sum': 1},
                'avg_cost': {'$avg': '$ps_supplycost'},
                'total_qty': {'$sum': '$ps_availqty'},
            }},
            {'$sort': {'part_count': -1}},
            {'$limit': 20},
        ])

    @query('group-10', 'Supplier group by nationkey')
    def _supplier_group_nationkey(self):
        return MongoAggregateQuery('supplier', [
            {'$group': {
                '_id': '$s_nationkey',
                'supplier_count': {'$sum': 1},
                'avg_balance': {'$avg': '$s_acctbal'},
            }},
            {'$sort': {'avg_balance': -1}},
        ])

    #endregion
    #region Embedded

    @query('embedded-0', 'orders with embedded lineitems shipdate filter')
    def _orders_embedded_shipdate(self):
        return MongoFindQuery('orders',
            filter={'lineitems.l_shipdate': {'$gte': self._param_date()}},
            projection={'o_orderkey': 1, 'o_custkey': 1, 'lineitems': {'$slice': 5}, '_id': 0},
            limit=self._param_limit(),
        )

    @query('embedded-1', 'part with embedded partsupp cost filter')
    def _part_embedded_partsupp_cost(self):
        return MongoFindQuery('part',
            filter={
                'partsupps': {
                    '$elemMatch': {
                        'ps_supplycost': {'$lt': self._param_float('cost', 10, 500)},
                        'ps_availqty': {'$gt': self._param_int('quantity', 100, 9000)},
                    }
                }
            },
            projection={'p_partkey': 1, 'p_name': 1, 'partsupps': {'$slice': 5}, '_id': 0},
            limit=self._param_limit(),
        )

    @query('embedded-2', 'orders unwind embedded lineitems and group by shipmode')
    def _orders_embedded_group_shipmode(self):
        return MongoAggregateQuery('orders', [
            {'$match': {'o_orderdate': {'$gte': self._param_date(1996, 1998)}}},
            {'$unwind': '$lineitems'},
            {'$group': {
                '_id': '$lineitems.l_shipmode',
                'count': {'$sum': 1},
                'sum_price': {'$sum': '$lineitems.l_extendedprice'},
                'avg_discount': {'$avg': '$lineitems.l_discount'},
            }},
            {'$sort': {'sum_price': -1}},
        ])

    @query('embedded-3', 'part unwind embedded partsupp and group by supplier')
    def _part_embedded_group_supplier(self):
        return MongoAggregateQuery('part', [
            {'$match': {'p_size': {'$gte': self._param_int('size_low', 1, 40)}}},
            {'$unwind': '$partsupps'},
            {'$group': {
                '_id': '$partsupps.ps_suppkey',
                'part_count': {'$sum': 1},
                'avg_supplycost': {'$avg': '$partsupps.ps_supplycost'},
                'total_availqty': {'$sum': '$partsupps.ps_availqty'},
            }},
            {'$sort': {'part_count': -1}},
            {'$limit': 30},
        ])

    @query('embedded-4', 'customer grouped by embedded nation and region')
    def _customer_group_embedded_geography(self):
        return MongoAggregateQuery('customer', [
            {'$match': {'c_acctbal': {'$gt': 0}}},
            {'$group': {
                '_id': {
                    'region': '$nation.region.r_name',
                    'nation': '$nation.n_name',
                },
                'count': {'$sum': 1},
                'avg_balance': {'$avg': '$c_acctbal'},
            }},
            {'$sort': {'avg_balance': -1}},
            {'$limit': 50},
        ])

    @query('embedded-5', 'supplier embedded geography filter and sort by balance')
    def _supplier_embedded_geography_balance(self):
        return MongoFindQuery('supplier',
            filter={
                'nation.region.r_name': self._param_choice('region', ['AFRICA', 'AMERICA', 'ASIA', 'EUROPE', 'MIDDLE EAST']),
                's_acctbal': {'$gt': self._param_int('balance', -1000, 10000)},
            },
            sort={'s_acctbal': -1},
            limit=self._param_limit(),
        )

    @query('embedded-6', 'orders embedded lineitems with elemMatch on mode, qty, and date')
    def _orders_embedded_elem_match(self):
        return MongoFindQuery('orders',
            filter={
                'lineitems': {
                    '$elemMatch': {
                        'l_shipmode': self._param_choice('shipmode', ['AIR', 'RAIL', 'SHIP', 'TRUCK', 'MAIL', 'FOB']),
                        'l_quantity': {'$gt': self._param_int('quantity', 1, 50)},
                        'l_shipdate': {'$gte': self._param_date(1996, 1998)},
                    }
                }
            },
            projection={'o_orderkey': 1, 'o_orderdate': 1, '_id': 0},
            limit=self._param_limit(),
        )

    @query('embedded-7', 'orders embedded lineitems revenue projection and group by order priority')
    def _orders_embedded_revenue_by_priority(self):
        return MongoAggregateQuery('orders', [
            {'$match': {'o_orderdate': {'$gte': self._param_date(1996, 1998)}}},
            {'$project': {
                'o_orderpriority': 1,
                'line_revenue': {
                    '$sum': {
                        '$map': {
                            'input': '$lineitems',
                            'as': 'li',
                            'in': {
                                '$multiply': ['$$li.l_extendedprice', {'$subtract': [1, '$$li.l_discount']}]
                            }
                        }
                    }
                }
            }},
            {'$group': {
                '_id': '$o_orderpriority',
                'orders': {'$sum': 1},
                'total_revenue': {'$sum': '$line_revenue'},
                'avg_revenue': {'$avg': '$line_revenue'},
            }},
            {'$sort': {'total_revenue': -1}},
        ])

    @query('embedded-8', 'part embedded partsupp unwind then cheapest per part')
    def _part_embedded_cheapest_supplier_per_part(self):
        return MongoAggregateQuery('part', [
            {'$match': {'p_size': {'$gte': self._param_int('size_threshold', 1, 40)}}},
            {'$unwind': '$partsupps'},
            {'$sort': {'partsupps.ps_supplycost': 1}},
            {'$group': {
                '_id': '$p_partkey',
                'part_name': {'$first': '$p_name'},
                'min_supplycost': {'$first': '$partsupps.ps_supplycost'},
                'supplier_key': {'$first': '$partsupps.ps_suppkey'},
            }},
            {'$limit': 100},
        ])

    @query('embedded-9', 'customer embedded region filter with segment grouping')
    def _customer_embedded_region_segment_group(self):
        return MongoAggregateQuery('customer', [
            {'$match': {
                'nation.region.r_name': self._param_choice('region', ['AFRICA', 'AMERICA', 'ASIA', 'EUROPE', 'MIDDLE EAST']),
                'c_acctbal': {'$gt': self._param_int('balance', -1000, 9000)},
            }},
            {'$group': {
                '_id': '$c_mktsegment',
                'customer_count': {'$sum': 1},
                'avg_balance': {'$avg': '$c_acctbal'},
            }},
            {'$sort': {'customer_count': -1}},
        ])

    #endregion
    #region Customer

    @query('customer-0', 'customer segment balance')
    def _customer_segment_balance_limit(self):
        return MongoFindQuery('customer',
            filter={
                'c_acctbal': {'$gt': self._param_int('balance', -1000, 9000)},
                'c_mktsegment': self._param_segment(),
            },
            sort={'c_acctbal': -1},
            limit=self._param_limit(),
        )

    @query('customer-1', 'customer balance only (varying selectivity)')
    def _customer_balance_only(self):
        return MongoFindQuery('customer',
            filter={'c_acctbal': {'$gt': self._param_int('balance', -1000, 9500)}},
            limit=self._param_limit(),
        )

    #endregion
    #region Part

    @query('part-0', 'part brand size filter')
    def _part_brand_size_limit(self):
        base = self._rng_int(1, 40)
        size_low = self._param('size_low', lambda: base)
        size_high = self._param('size_high', lambda: base + self._rng_int(5, 20))

        return MongoFindQuery('part',
            filter={
                'p_size': {'$gte': size_low, '$lte': size_high},
                'p_brand': self._param_brand()
            },
            sort={'p_retailprice': -1},
            limit=self._param_limit(),
        )

    #endregion
    #region Supplier
    # (small collection scans)

    @query('supplier-0', 'supplier balance filter')
    def _supplier_balance_sort(self):
        return MongoFindQuery('supplier',
            filter={'s_acctbal': {'$gt': self._param_int('balance', 0, 10000)}},
            sort={'s_acctbal': -1},
        )

    @query('supplier-1', 'supplier balance range')
    def _supplier_balance_range(self):
        base = self._rng_int(-1000, 5000)
        balance_low = self._param('balance_low', lambda: base)
        balance_high = self._param('balance_high', lambda: base + self._rng_int(500, 5000))

        return MongoFindQuery('supplier',
            filter={'s_acctbal': {'$gte': balance_low, '$lte': balance_high}},
        )

    @query('supplier-2', 'supplier nation filter')
    def _supplier_nation(self):
        return MongoFindQuery('supplier',
            filter={'s_nationkey': self._param_nationkey()},
        )

    #endregion
    #region Full scan

    @query('full-scan-0', 'nation full scan')
    def _nation_full(self):
        return MongoFindQuery('nation')

    @query('full-scan-1', 'region full scan')
    def _region_full(self):
        return MongoFindQuery('region')

    @query('full-scan-2', 'Customer full scan (30K rows)')
    def _customer_full(self):
        return MongoFindQuery('customer')

    @query('full-scan-3', 'Partsupp full scan (160K rows)')
    def _partsup_full(self):
        return MongoFindQuery('partsupp')

    @query('full-scan-4', 'Supplier full scan (2K rows)')
    def _supplier_full(self):
        return MongoFindQuery('supplier')

    #endregion
    #region Skip

    @query('skip-0', 'lineitem + skip + limit')
    def _lineitem_skip_limit(self):
        return MongoFindQuery('lineitem',
            filter={'l_shipdate': {'$gte': self._param_date()}},
            limit=self._param_limit(),
            skip=self._param_skip(),
        )

    @query('skip-1', 'Orders deep pagination skip = 5000')
    def _orders_deep_pagination(self):
        return MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 100000}},
            sort={'o_totalprice': -1},
            skip=5000,
            limit=50,
        )

    @query('skip-2', 'Lineitem deep skip = 10000')
    def _lineitem_deep_skip(self):
        return MongoFindQuery('lineitem',
            filter={'l_shipdate': {'$gte': self._param_date(1997, 1997)}},
            skip=10000,
            limit=100,
        )

    @query('skip-3', 'Customer skip = 15000 sorted')
    def _customer_skip_sorted(self):
        return MongoFindQuery('customer',
            filter={},
            sort={'c_acctbal': -1},
            skip=15000,
            limit=50,
        )

    #endregion
    #region Projection-only

    @query('projection-0', 'customer projection lookup')
    def _customer_projection(self):
        return MongoFindQuery('customer',
            filter={'c_custkey': self._param_custkey()},
            projection={'c_name': 1, 'c_acctbal': 1, '_id': 0},
        )

    #endregion
    #region Complex aggregation + $group + $sort

    @query('agg-complex-0', 'TPC-H Q1 variants with varying date ranges (from narrow to full-table scan)')
    def _tpch_lineitem(self):
        return MongoAggregateQuery('lineitem', [
            {'$match': {'l_shipdate': {'$gte': self._param_date()}}},
            {'$group': {
                '_id': {'flag': '$l_returnflag', 'status': '$l_linestatus'},
                'sum_qty': {'$sum': '$l_quantity'},
                'sum_price': {'$sum': '$l_extendedprice'},
                'avg_disc': {'$avg': '$l_discount'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'sum_price': -1}},
        ])

    @query('agg-complex-1', 'TPC-H Q1 with $lte (matching evaluation style)')
    def _tpch_lineitem_lte(self):
        return MongoAggregateQuery('lineitem', [
            {'$match': {'l_shipdate': {'$lte': self._param_date()}}},
            {'$group': {
                '_id': {'rf': '$l_returnflag', 'ls': '$l_linestatus'},
                'sum_qty': {'$sum': '$l_quantity'},
                'sum_price': {'$sum': '$l_extendedprice'},
                'avg_disc': {'$avg': '$l_discount'},
                'count': {'$sum': 1},
            }},
            {'$sort': {'_id': 1}},
        ])

    @query('agg-complex-2', 'Simple lineitem aggregation (count + sum only, less accumulators)')
    def _tpch_lineitem_simple(self):
        return MongoAggregateQuery('lineitem', [
            {'$match': {'l_shipdate': {'$gte': self._param_date()}}},
            {'$group': {
                '_id': '$l_returnflag',
                'count': {'$sum': 1},
            }},
        ])

    @query('agg-complex-3', 'Add partsupp queries (multiple patterns including just cost filter)')
    def _tpch_partsupp(self):
        return MongoFindQuery('partsupp',
            filter={
                'ps_supplycost': {'$lt': self._param_float('cost', 10, 500)},
                'ps_availqty': {'$gt': self._param_int('quantity', 100, 9000)},
            },
            sort={'ps_supplycost': 1},
            limit=self._param_limit(),
        )

    @query('agg-complex-4', 'Partsupp with just cost filter + sort (matches evaluation)')
    def _tpch_partsupp_lte(self):
        return MongoFindQuery('partsupp',
            filter={'ps_supplycost': {'$lt': self._param_float('cost', 10, 1000)}},
            sort={'ps_supplycost': 1},
            limit=self._param_limit(),
        )

    @query('agg-complex-5', 'Supplier without sort (matches evaluation pattern)')
    def _tpch_supplier(self):
        return MongoFindQuery('supplier',
            filter={'s_acctbal': {'$gt': self._param_int('balance', -1000, 10000)}},
        )

    #endregion
    #region Predicates
    # ($or, $and, $ne, $nin, $exists, $regex)

    # $or across different fields (training never uses $or in find)

    @query('predicate-0', 'Orders $or extreme prices')
    def _orders_extreme_prices_or(self):
        return MongoFindQuery('orders',
            filter={'$or': [
                {'o_totalprice': {'$lt': 5000}},
                {'o_totalprice': {'$gt': 450000}},
            ]},
            limit=200,
        )

    @query('predicate-1', 'Customer $or segment + balance')
    def _customer_segment_balance(self):
        return MongoFindQuery('customer',
            filter={'$or': [
                {'c_mktsegment': 'BUILDING', 'c_acctbal': {'$gt': 9000}},
                {'c_mktsegment': 'FURNITURE', 'c_acctbal': {'$lt': -500}},
            ]},
        )

    @query('predicate-2', 'Lineitem $or returnflag|shipmode + quantity')
    def _lineitem_returnflag(self):
        return MongoFindQuery('lineitem',
            filter={'$or': [
                {'l_returnflag': 'R'},
                {'l_shipmode': 'AIR', 'l_quantity': {'$gt': 45}},
            ]},
            limit=500,
        )

    # $and with many predicates (3+ field compound filter)

    @query('predicate-3', 'Lineitem $and 4-pred (TPC-H Q6 style)')
    def _lineitem_and_4pred(self):
        start_date = self._rng_date(1995, 1996)
        end_date = start_date + timedelta(days=365)

        return MongoFindQuery('lineitem',
            filter={'$and': [
                {'l_shipdate': {'$gte': self._param('start_date', lambda: start_date)}},
                {'l_shipdate': {'$lt': self._param('end_date', lambda: end_date)}},
                {'l_discount': {'$gte': 0.05, '$lte': 0.07}},
                {'l_quantity': {'$lt': 24}},
            ]},
        )

    @query('predicate-4', 'Orders $and date + priority + price')
    def _orders_date_priority_price(self):
        start_date = self._rng_date(1994, 1995)
        end_date = start_date + timedelta(days=365)

        return MongoFindQuery('orders',
            filter={'$and': [
                {'o_orderdate': {'$gte': self._param('start_date', lambda: start_date)}},
                {'o_orderdate': {'$lt': self._param('end_date', lambda: end_date)}},
                {'o_orderpriority': {'$in': ['1-URGENT', '2-HIGH']}},
                {'o_totalprice': {'$gt': 100000}},
            ]},
            limit=100,
        )

    # $ne operator (never in training)

    @query('predicate-5', 'Orders status $ne F')
    def _orders_status_ne_F(self):
        return MongoFindQuery('orders',
            filter={'o_orderstatus': {'$ne': 'F'}},
            limit=200,
        )

    @query('predicate-6', 'Lineitem $ne returnflag + shipmode')
    def _lineitem_ne_returnflag_shipmode(self):
        return MongoFindQuery('lineitem',
            filter={'l_returnflag': {'$ne': 'N'}, 'l_shipmode': {'$ne': 'TRUCK'}},
            limit=300,
        )

    # $nin operator

    @query('predicate-7', 'Lineitem shipmode $nin 3 values')
    def _lineitem_shipmode_nin(self):
        return MongoFindQuery('lineitem',
            filter={'l_shipmode': {'$nin': ['AIR', 'MAIL', 'RAIL']}},
            limit=500,
        )

    @query('predicate-8', 'Orders priority $nin urgent/high')
    def _orders_priority_nin(self):
        return MongoFindQuery('orders',
            filter={'o_orderpriority': {'$nin': ['1-URGENT', '2-HIGH']}},
            limit=200,
        )

    # $exists (structural predicate — never in training)

    @query('predicate-9', 'Customer $exists comment + high balance')
    def _customer_high_balance(self):
        return MongoFindQuery('customer',
            filter={
                'c_comment': {'$exists': True},
                'c_acctbal': {'$gt': 8000},
            },
        )

    # $regex (never in training)

    @query('predicate-10', 'Customer $regex name prefix')
    def _customer_regex_name(self):
        return MongoFindQuery('customer',
            filter={'c_name': {'$regex': '^Customer#00001'}},
            limit=50,
        )

    @query('predicate-11', 'Part $regex type ends BRASS')
    def _part_regex_type(self):
        return MongoFindQuery('part',
            filter={'p_type': {'$regex': 'BRASS$'}},
        )

    @query('predicate-12', 'Part $regex name contains green')
    def _part_regex_name(self):
        return MongoFindQuery('part',
            filter={'p_name': {'$regex': 'green'}},
        )

    # Nested $or inside $and

    @query('predicate-13', 'Lineitem nested $and/$or (TPC-H Q19 style)')
    def _lineitem_nested_and_or(self):
        return MongoFindQuery('lineitem',
            filter={'$and': [
                {'l_shipdate': {'$gte': self._param_date(1995, 1996)}},
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
        )

    #endregion
    #region Selective

    # Near-empty results (very tight filter)

    @query('selective-0', 'Orders extreme high price (near empty)')
    def _orders_extreme_prices(self):
        return MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 490000}},
        )

    @query('selective-1', 'Customer balance > 9990 (near empty)')
    def _customer_balance_nin(self):
        return MongoFindQuery('customer',
            filter={'c_acctbal': {'$gt': 9990}},
        )

    @query('selective-2', 'Lineitem exact quantity = 1 disc = 0 tax = 0 (near empty)')
    def _lineitem_exact_qty_disc_tax(self):
        return MongoFindQuery('lineitem',
            filter={'l_quantity': 1.0, 'l_discount': 0.0, 'l_tax': 0.0},
        )

    # Near-full-table scans (very permissive filter)

    @query('permissive-0', 'Lineitem quantity > 0 (full table, 1.2M rows)')
    def _lineitem_quantity_positive(self):
        return MongoFindQuery('lineitem',
            filter={'l_quantity': {'$gt': 0}},
        )

    @query('permissive-1', 'Orders price > 0 (full table, 300K rows)')
    def _orders_price_positive(self):
        return MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 0}},
        )

    # Unlimited scans with sort on medium table (no limit!)

    @query('unlimited-0', 'Customer balance > 0 sorted NO LIMIT')
    def _customer_balance_sorted(self):
        return MongoFindQuery('customer',
            filter={'c_acctbal': {'$gt': 0}},
            sort={'c_acctbal': -1},
        )

    @query('unlimited-1', 'Orders 1997+ sorted by price NO LIMIT')
    def _orders_price_sorted(self):
        return MongoFindQuery('orders',
            filter={'o_orderdate': {'$gte': self._param_date(1997, 1997)}},
            sort={'o_totalprice': -1},
        )

    #endregion
    #region Unseen

    @query('unseen-0', 'Lineitem commitdate < 1993-06 LIMIT 200')
    def _lineitem_commitdate(self):
        return MongoFindQuery('lineitem',
            filter={'l_commitdate': {'$lt': self._param_date(1993, 1993)}},
            limit=200,
        )

    @query('unseen-1', 'Lineitem receiptdate >= 1998-06')
    def _lineitem_receiptdate(self):
        return MongoFindQuery('lineitem',
            filter={'l_receiptdate': {'$gte': self._param_date(1998, 1998)}},
            limit=500,
        )

    @query('unseen-2', 'Lineitem 3-date cross filter')
    def _lineitem_cross_date_filter(self):
        return MongoFindQuery('lineitem',
            filter={
                'l_shipdate': {'$gte': datetime(1995, 1, 1)},
                'l_commitdate': {'$lt': datetime(1995, 3, 1)},
                'l_receiptdate': {'$gte': datetime(1995, 2, 1)},
            },
            limit=200,
        )

    @query('unseen-3', 'Orders status filter')
    def _orders_status_filter(self):
        return MongoFindQuery('orders',
            filter={'o_orderstatus': self._param_order_status()},
            limit=500,
        )

    @query('unseen-4', 'Orders exact clerk lookup')
    def _orders_exact_clerk_lookup(self):
        return MongoFindQuery('orders',
            filter={'o_clerk': 'Clerk#000000951'},
        )

    @query('unseen-5', 'Lineitem shipinstruct + mode (string equality combo)')
    def _lineitem_shipinstruct_mode(self):
        return MongoFindQuery('lineitem',
            filter={'l_shipinstruct': 'DELIVER IN PERSON', 'l_shipmode': 'AIR'},
            limit=300,
        )

    @query('unseen-6', 'Part small containers size <= 5')
    def _part_small_containers(self):
        return MongoFindQuery('part',
            filter={'p_container': {'$in': ['SM CASE', 'SM BOX', 'SM PACK', 'SM PKG']}, 'p_size': {'$lte': 5}},
        )

    @query('unseen-7', 'Part Mfgr#3 ECONOMY types')
    def _part_mfgr_regex(self):
        return MongoFindQuery('part',
            filter={'p_mfgr': 'Manufacturer#3', 'p_type': {'$regex': '^ECONOMY'}},
        )

    @query('unseen-8', 'Customer by nationkey')
    def _customer_by_nationkey(self):
        return MongoFindQuery('customer',
            filter={'c_nationkey': self._param_nationkey()},
        )

    @query('unseen-9', 'Partsupp very low avail quantity < 100')
    def _partsupp_low_availqty(self):
        return MongoFindQuery('partsupp',
            filter={'ps_availqty': {'$lt': 100}},
        )

    @query('unseen-10', 'Partsupp high quantity + cost')
    def _partsupp_high_qty_cost(self):
        return MongoFindQuery('partsupp',
            filter={'ps_availqty': {'$gt': 9500}, 'ps_supplycost': {'$gt': 900}},
        )

    #endregion
    #region $in on large lists

    # Large $in list on indexed field

    @query('in-indexed-0', 'Orders 50-key $in lookup')
    def _orders_50_key_in_lookup(self):
        return MongoFindQuery('orders',
            filter={'o_orderkey': {'$in': self._param_orderkeys(50)}},
        )

    @query('in-indexed-1', 'Orders 200-key $in lookup')
    def _orders_200_key_in_lookup(self):
        return MongoFindQuery('orders',
            filter={'o_orderkey': {'$in': self._param_orderkeys(200)}},
        )

    # $in on non-indexed field (string values)

    @query('in-other-0', 'Lineitem shipmode $in [AIR,RAIL]')
    def _lineitem_shipmode_in(self):
        return MongoFindQuery('lineitem',
            filter={'l_shipmode': {'$in': ['AIR', 'RAIL']}},
            limit=1000,
        )

    @query('in-other-1', 'Part 9-brand $in lookup')
    def _part_9_brand_in(self):
        return MongoFindQuery('part',
            filter={'p_brand': {'$in': self._param_brands()}},
        )

    #endregion
    #region Novel aggregations

    # G1. $count stage
    @query('count-0', 'Lineitem $count AIR 1997+')
    def _lineitem_count_air(self):
        return MongoAggregateQuery('lineitem', [
            {'$match': {
                'l_shipmode': 'AIR',
                'l_shipdate': {'$gte': self._param_date(1997, 1997)},
            }},
            {'$count': 'total'},
        ])

    @query('count-1', 'Orders $count status = P')
    def _orders_count_status(self):
        return MongoAggregateQuery('orders', [
            {'$match': {'o_orderstatus': 'P'}},
            {'$count': 'total'},
        ])



    #endregion
    #region Limit

    @query('limit-0', 'Lineitem top-1 most expensive 1995+')
    def _lineitem_top_1_expensive(self):
        return MongoFindQuery('lineitem',
            filter={'l_shipdate': {'$gte': self._param_date(1995, 1995)}},
            sort={'l_extendedprice': -1},
            limit=1,
        )

    @query('limit-1', 'Lineitem top-5000 most expensive 1995+')
    def _lineitem_top_5000_expensive(self):
        return MongoFindQuery('lineitem',
            filter={'l_shipdate': {'$gte': self._param_date(1995, 1995)}},
            sort={'l_extendedprice': -1},
            limit=5000,
        )

    @query('limit-2', 'Orders earliest order price > 100K (limit 1)')
    def _orders_earliest_high_value(self):
        return MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 100000}},
            sort={'o_orderdate': 1},
            limit=1,
        )

    @query('limit-3', 'Orders price > 100K sorted by date limit 10000')
    def _orders_high_value_sorted(self):
        return MongoFindQuery('orders',
            filter={'o_totalprice': {'$gt': 100000}},
            sort={'o_orderdate': 1},
            limit=10000,
        )

    #endregion
