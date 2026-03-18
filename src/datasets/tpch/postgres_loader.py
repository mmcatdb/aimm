from typing_extensions import override
from common.loaders.postgres_loader import PostgresLoader, ColumnSchema, IndexSchema

class TpchPostgresLoader(PostgresLoader):
    @override
    def name(self):
        return 'TPC-H'

    # TODO Add not null constraints.

    @override
    def _get_schemas(self):
        return get_postgres_tpch_schemas()

    @override
    def _get_indexes(self):
        return [
            IndexSchema('nation', [ 'n_regionkey' ]),
            IndexSchema('customer', [ 'c_nationkey' ]),
            IndexSchema('orders', [ 'o_custkey' ]),
            IndexSchema('supplier', [ 's_nationkey' ]),
            IndexSchema('partsupp', [ 'ps_partkey' ]),
            IndexSchema('partsupp', [ 'ps_suppkey' ]),
            IndexSchema('lineitem', [ 'l_orderkey' ]),
            IndexSchema('lineitem', [ 'l_partkey' ]),
            IndexSchema('lineitem', [ 'l_suppkey' ]),

            # These fields are by far the most common in queries:
            IndexSchema('orders', [ 'o_orderdate' ]),
            IndexSchema('lineitem', [ 'l_shipdate' ]),
        ]

def get_postgres_tpch_schemas() -> dict[str, list[ColumnSchema]]:
    region = [
        ColumnSchema('r_regionkey', 'INTEGER', primary_key=True),
        ColumnSchema('r_name', 'CHAR(25)'),
        ColumnSchema('r_comment', 'VARCHAR(152)'),
    ]

    nation = [
        ColumnSchema('n_nationkey', 'INTEGER', primary_key=True),
        ColumnSchema('n_name', 'CHAR(25)'),
        ColumnSchema('n_regionkey', 'INTEGER', references='region(r_regionkey)'),
        ColumnSchema('n_comment', 'VARCHAR(152)'),
    ]

    customer = [
        ColumnSchema('c_custkey', 'INTEGER', primary_key=True),
        ColumnSchema('c_name', 'VARCHAR(25)'),
        ColumnSchema('c_address', 'VARCHAR(40)'),
        ColumnSchema('c_nationkey', 'INTEGER', references='nation(n_nationkey)'),
        ColumnSchema('c_phone', 'CHAR(15)'),
        ColumnSchema('c_acctbal', 'DECIMAL(15,2)'),
        ColumnSchema('c_mktsegment', 'CHAR(10)'),
        ColumnSchema('c_comment', 'VARCHAR(117)'),
    ]

    orders = [
        ColumnSchema('o_orderkey', 'INTEGER', primary_key=True),
        ColumnSchema('o_custkey', 'INTEGER', references='customer(c_custkey)'),
        ColumnSchema('o_orderstatus', 'CHAR(1)'),
        ColumnSchema('o_totalprice', 'DECIMAL(15,2)'),
        ColumnSchema('o_orderdate', 'DATE'),
        ColumnSchema('o_orderpriority', 'CHAR(15)'),
        ColumnSchema('o_clerk', 'CHAR(15)'),
        ColumnSchema('o_shippriority', 'INTEGER'),
        ColumnSchema('o_comment', 'VARCHAR(79)'),
    ]

    part = [
        ColumnSchema('p_partkey', 'INTEGER', primary_key=True),
        ColumnSchema('p_name', 'VARCHAR(55)'),
        ColumnSchema('p_mfgr', 'CHAR(25)'),
        ColumnSchema('p_brand', 'CHAR(10)'),
        ColumnSchema('p_type', 'VARCHAR(25)'),
        ColumnSchema('p_size', 'INTEGER'),
        ColumnSchema('p_container', 'CHAR(10)'),
        ColumnSchema('p_retailprice', 'DECIMAL(15,2)'),
        ColumnSchema('p_comment', 'VARCHAR(23)'),
    ]

    supplier = [
        ColumnSchema('s_suppkey', 'INTEGER', primary_key=True),
        ColumnSchema('s_name', 'CHAR(25)'),
        ColumnSchema('s_address', 'VARCHAR(40)'),
        ColumnSchema('s_nationkey', 'INTEGER', references='nation(n_nationkey)'),
        ColumnSchema('s_phone', 'CHAR(15)'),
        ColumnSchema('s_acctbal', 'DECIMAL(15,2)'),
        ColumnSchema('s_comment', 'VARCHAR(101)'),
    ]

    partsupp = [
        ColumnSchema('ps_partkey', 'INTEGER', primary_key=True, references='part(p_partkey)'),
        ColumnSchema('ps_suppkey', 'INTEGER', primary_key=True, references='supplier(s_suppkey)'),
        ColumnSchema('ps_availqty', 'INTEGER'),
        ColumnSchema('ps_supplycost', 'DECIMAL(15,2)'),
        ColumnSchema('ps_comment', 'VARCHAR(255)'),
    ]

    lineitem = [
        ColumnSchema('l_orderkey', 'INTEGER', primary_key=True, references='orders(o_orderkey)'),
        ColumnSchema('l_partkey', 'INTEGER', references='part(p_partkey)'),
        ColumnSchema('l_suppkey', 'INTEGER', references='supplier(s_suppkey)'),
        ColumnSchema('l_linenumber', 'INTEGER', primary_key=True),
        ColumnSchema('l_quantity', 'DECIMAL(15,2)'),
        ColumnSchema('l_extendedprice', 'DECIMAL(15,2)'),
        ColumnSchema('l_discount', 'DECIMAL(15,2)'),
        ColumnSchema('l_tax', 'DECIMAL(15,2)'),
        ColumnSchema('l_returnflag', 'CHAR(1)'),
        ColumnSchema('l_linestatus', 'CHAR(1)'),
        ColumnSchema('l_shipdate', 'DATE'),
        ColumnSchema('l_commitdate', 'DATE'),
        ColumnSchema('l_receiptdate', 'DATE'),
        ColumnSchema('l_shipinstruct', 'CHAR(25)'),
        ColumnSchema('l_shipmode', 'CHAR(10)'),
        ColumnSchema('l_comment', 'VARCHAR(44)'),
    ]

    # Custom tables (not part of the original TPC-H schema)

    knows = [
        ColumnSchema('k_custkey1', 'INTEGER', primary_key=True, references='customer(c_custkey)'),
        ColumnSchema('k_custkey2', 'INTEGER', primary_key=True, references='customer(c_custkey)'),
        ColumnSchema('k_startdate', 'DATE'),
        ColumnSchema('k_source', 'VARCHAR(20)'), # e.g. 'organic', 'social_media', etc.
        ColumnSchema('k_comment', 'VARCHAR(255)'), # optional free-form comment about the relationship
        ColumnSchema('k_strength', 'DECIMAL(3,2)'), # Strength of the relationship (0.00 to 1.00)
    ]

    return {
        'region': region,
        'nation': nation,
        'customer': customer,
        'orders': orders,
        'part': part,
        'supplier': supplier,
        'partsupp': partsupp,
        'lineitem': lineitem,
        'knows': knows,
    }
