from typing_extensions import override
from common.loaders.mongo_loader import MongoLoader, MongoPostgresBuilder
from common.daos.mongo_dao import IndexSchema
from datasets.tpch.postgres_loader import get_postgres_tpch_schemas

class TpchMongoLoader(MongoLoader):
    @override
    def name(self):
        return 'TPC-H'

    @override
    def _get_schemas(self):
        b = MongoPostgresBuilder.create(get_postgres_tpch_schemas())

        nation = b.document('nation', {
            'key':     'n_nationkey',
            'name':    'n_name',
            'comment': 'n_comment',
            'region': b.nested('region', {
                'key':     'r_regionkey',
                'name':    'r_name',
                'comment': 'r_comment',
            }, parent_join='n_regionkey', child_join='r_regionkey'),
        })

        order = b.document('orders', {
            'key':           'o_orderkey',
            'custkey':       'o_custkey',
            'orderstatus':   'o_orderstatus',
            'totalprice':    'o_totalprice',
            'orderdate':     'o_orderdate',
            'orderpriority': 'o_orderpriority',
            'clerk':         'o_clerk',
            'shippriority':  'o_shippriority',
            'comment':       'o_comment',
            'lines': b.nested('lineitem', {
                'partkey':       'l_partkey',
                'suppkey':       'l_suppkey',
                'linenumber':    'l_linenumber',
                'quantity':      'l_quantity',
                'extendedprice': 'l_extendedprice',
                'discount':      'l_discount',
                'tax':           'l_tax',
                'returnflag':    'l_returnflag',
                'linestatus':    'l_linestatus',
                'shipdate':      'l_shipdate',
                'commitdate':    'l_commitdate',
                'receiptdate':   'l_receiptdate',
                'shipinstruct':  'l_shipinstruct',
                'shipmode':      'l_shipmode',
                'comment':       'l_comment',
            }, parent_join='o_orderkey', child_join='l_orderkey', is_array=True),
        })

        return [
            nation,
            order,
        ]

    @override
    def _get_indexes(self):
        return [
            # IndexSchema('region', [ 'r_regionkey' ], is_unique=True),
            IndexSchema('nation', [ 'key' ], is_unique=True),
            # IndexSchema('customer', [ 'c_custkey' ], is_unique=True),
            # IndexSchema('orders', [ 'o_orderkey' ], is_unique=True),
            # IndexSchema('part', [ 'p_partkey' ], is_unique=True),
            # IndexSchema('supplier', [ 's_suppkey' ], is_unique=True),
            # IndexSchema('partsupp', [ 'ps_partkey', 'ps_suppkey' ], is_unique=True),
            # IndexSchema('lineitem', [ 'l_orderkey', 'l_linenumber' ], is_unique=True),

            # TODO some other indexes
        ]

