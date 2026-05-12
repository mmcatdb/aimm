from typing_extensions import override
from core.loaders.mongo_loader import MongoLoader, MongoPostgresBuilder, MongoIndex
from ...postgres.tpch.loader import get_postgres_tpch_kinds

def export():
    return MongoTpchLoader()

class MongoTpchLoader(MongoLoader):

    @override
    def _get_csv_kinds(self):
        b = MongoPostgresBuilder.create(get_postgres_tpch_kinds())

        customer = b.document('customer', {
            'c_custkey': 'c_custkey',
            'c_name': 'c_name',
            'c_address': 'c_address',
            'c_nationkey': 'c_nationkey',
            'c_phone': 'c_phone',
            'c_acctbal': 'c_acctbal',
            'c_mktsegment': 'c_mktsegment',
            'c_comment': 'c_comment',
            'nation': b.nested('nation', {
                'n_nationkey': 'n_nationkey',
                'n_name': 'n_name',
                'n_regionkey': 'n_regionkey',
                'n_comment': 'n_comment',
                'region': b.nested('region', {
                    'r_regionkey': 'r_regionkey',
                    'r_name': 'r_name',
                    'r_comment': 'r_comment',
                }, parent_join='n_regionkey', child_join='r_regionkey'),
            }, parent_join='c_nationkey', child_join='n_nationkey'),
        })

        supplier = b.document('supplier', {
            's_suppkey': 's_suppkey',
            's_name': 's_name',
            's_address': 's_address',
            's_nationkey': 's_nationkey',
            's_phone': 's_phone',
            's_acctbal': 's_acctbal',
            's_comment': 's_comment',
            'nation': b.nested('nation', {
                'n_nationkey': 'n_nationkey',
                'n_name': 'n_name',
                'n_regionkey': 'n_regionkey',
                'n_comment': 'n_comment',
                'region': b.nested('region', {
                    'r_regionkey': 'r_regionkey',
                    'r_name': 'r_name',
                    'r_comment': 'r_comment',
                }, parent_join='n_regionkey', child_join='r_regionkey'),
            }, parent_join='s_nationkey', child_join='n_nationkey'),
        })

        orders = b.document('orders', {
            'o_orderkey': 'o_orderkey',
            'o_custkey': 'o_custkey',
            'o_orderstatus': 'o_orderstatus',
            'o_totalprice': 'o_totalprice',
            'o_orderdate': 'o_orderdate',
            'o_orderpriority': 'o_orderpriority',
            'o_clerk': 'o_clerk',
            'o_shippriority': 'o_shippriority',
            'o_comment': 'o_comment',
            'lineitems': b.nested('lineitem', {
                'l_orderkey': 'l_orderkey',
                'l_partkey': 'l_partkey',
                'l_suppkey': 'l_suppkey',
                'l_linenumber': 'l_linenumber',
                'l_quantity': 'l_quantity',
                'l_extendedprice': 'l_extendedprice',
                'l_discount': 'l_discount',
                'l_tax': 'l_tax',
                'l_returnflag': 'l_returnflag',
                'l_linestatus': 'l_linestatus',
                'l_shipdate': 'l_shipdate',
                'l_commitdate': 'l_commitdate',
                'l_receiptdate': 'l_receiptdate',
                'l_shipinstruct': 'l_shipinstruct',
                'l_shipmode': 'l_shipmode',
                'l_comment': 'l_comment',
            }, parent_join='o_orderkey', child_join='l_orderkey', is_array=True),
        })

        part = b.document('part', {
            'p_partkey': 'p_partkey',
            'p_name': 'p_name',
            'p_mfgr': 'p_mfgr',
            'p_brand': 'p_brand',
            'p_type': 'p_type',
            'p_size': 'p_size',
            'p_container': 'p_container',
            'p_retailprice': 'p_retailprice',
            'p_comment': 'p_comment',
            'partsupps': b.nested('partsupp', {
                'ps_partkey': 'ps_partkey',
                'ps_suppkey': 'ps_suppkey',
                'ps_availqty': 'ps_availqty',
                'ps_supplycost': 'ps_supplycost',
                'ps_comment': 'ps_comment',
            }, parent_join='p_partkey', child_join='ps_partkey', is_array=True),
        })

        return [
            b.plain_copy('region'),
            b.plain_copy('nation'),
            customer,
            orders,
            part,
            supplier,
            b.plain_copy('partsupp'),
            b.plain_copy('lineitem'),
        ]

    @override
    def _get_constraints(self):
        return [
            MongoIndex('region', [ 'r_regionkey' ], is_unique=True),
            MongoIndex('nation', [ 'n_nationkey' ], is_unique=True),
            MongoIndex('customer', [ 'c_custkey' ], is_unique=True),
            MongoIndex('customer', [ 'c_nationkey' ]),
            MongoIndex('customer', [ 'nation.n_nationkey' ]),
            MongoIndex('customer', [ 'nation.region.r_regionkey' ]),
            MongoIndex('orders', [ 'o_orderkey' ], is_unique=True),
            MongoIndex('orders', [ 'o_custkey' ]),
            MongoIndex('orders', [ 'o_orderdate' ]),
            MongoIndex('orders', [ 'lineitems.l_shipdate' ]),
            MongoIndex('part', [ 'p_partkey' ], is_unique=True),
            MongoIndex('part', [ 'partsupps.ps_suppkey' ]),
            MongoIndex('supplier', [ 's_suppkey' ], is_unique=True),
            MongoIndex('supplier', [ 's_nationkey' ]),
            MongoIndex('supplier', [ 'nation.n_nationkey' ]),
            MongoIndex('supplier', [ 'nation.region.r_regionkey' ]),
            MongoIndex('partsupp', [ 'ps_partkey', 'ps_suppkey' ], is_unique=True),
            MongoIndex('partsupp', [ 'ps_partkey' ]),
            MongoIndex('partsupp', [ 'ps_suppkey' ]),
            MongoIndex('lineitem', [ 'l_orderkey', 'l_linenumber' ], is_unique=True),
            MongoIndex('lineitem', [ 'l_orderkey' ]),
            MongoIndex('lineitem', [ 'l_partkey' ]),
            MongoIndex('lineitem', [ 'l_suppkey' ]),
            MongoIndex('lineitem', [ 'l_shipdate' ]),
        ]
