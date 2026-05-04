from typing_extensions import override
from core.loaders.postgres_loader import PostgresLoader, PostgresColumn, PostgresIndex

def export():
    return PostgresTpchLoader()

class PostgresTpchLoader(PostgresLoader):

    # TODO Add not null constraints.

    @override
    def _get_kinds(self):
        return get_postgres_tpch_kinds()

    @override
    def _get_constraints(self):
        return [
            PostgresIndex('nation', [ 'n_regionkey' ]),
            PostgresIndex('customer', [ 'c_nationkey' ]),
            PostgresIndex('orders', [ 'o_custkey' ]),
            PostgresIndex('supplier', [ 's_nationkey' ]),
            PostgresIndex('partsupp', [ 'ps_partkey' ]),
            PostgresIndex('partsupp', [ 'ps_suppkey' ]),
            PostgresIndex('lineitem', [ 'l_orderkey' ]),
            PostgresIndex('lineitem', [ 'l_partkey' ]),
            PostgresIndex('lineitem', [ 'l_suppkey' ]),

            # These fields are by far the most common in queries:
            PostgresIndex('orders', [ 'o_orderdate' ]),
            PostgresIndex('lineitem', [ 'l_shipdate' ]),
        ]

def get_postgres_tpch_kinds() -> dict[str, list[PostgresColumn]]:
    region = [
        PostgresColumn('r_regionkey', 'INTEGER', primary_key=True),
        PostgresColumn('r_name', 'CHAR(25)'),
        PostgresColumn('r_comment', 'VARCHAR(152)'),
    ]

    nation = [
        PostgresColumn('n_nationkey', 'INTEGER', primary_key=True),
        PostgresColumn('n_name', 'CHAR(25)'),
        PostgresColumn('n_regionkey', 'INTEGER', references='region(r_regionkey)'),
        PostgresColumn('n_comment', 'VARCHAR(152)'),
    ]

    customer = [
        PostgresColumn('c_custkey', 'INTEGER', primary_key=True),
        PostgresColumn('c_name', 'VARCHAR(25)'),
        PostgresColumn('c_address', 'VARCHAR(40)'),
        PostgresColumn('c_nationkey', 'INTEGER', references='nation(n_nationkey)'),
        PostgresColumn('c_phone', 'CHAR(15)'),
        PostgresColumn('c_acctbal', 'DECIMAL(15,2)'),
        PostgresColumn('c_mktsegment', 'CHAR(10)'),
        PostgresColumn('c_comment', 'VARCHAR(117)'),
    ]

    orders = [
        PostgresColumn('o_orderkey', 'INTEGER', primary_key=True),
        PostgresColumn('o_custkey', 'INTEGER', references='customer(c_custkey)'),
        PostgresColumn('o_orderstatus', 'CHAR(1)'),
        PostgresColumn('o_totalprice', 'DECIMAL(15,2)'),
        PostgresColumn('o_orderdate', 'DATE'),
        PostgresColumn('o_orderpriority', 'CHAR(15)'),
        PostgresColumn('o_clerk', 'CHAR(15)'),
        PostgresColumn('o_shippriority', 'INTEGER'),
        PostgresColumn('o_comment', 'VARCHAR(79)'),
    ]

    part = [
        PostgresColumn('p_partkey', 'INTEGER', primary_key=True),
        PostgresColumn('p_name', 'VARCHAR(55)'),
        PostgresColumn('p_mfgr', 'CHAR(25)'),
        PostgresColumn('p_brand', 'CHAR(10)'),
        PostgresColumn('p_type', 'VARCHAR(25)'),
        PostgresColumn('p_size', 'INTEGER'),
        PostgresColumn('p_container', 'CHAR(10)'),
        PostgresColumn('p_retailprice', 'DECIMAL(15,2)'),
        PostgresColumn('p_comment', 'VARCHAR(23)'),
    ]

    supplier = [
        PostgresColumn('s_suppkey', 'INTEGER', primary_key=True),
        PostgresColumn('s_name', 'CHAR(25)'),
        PostgresColumn('s_address', 'VARCHAR(40)'),
        PostgresColumn('s_nationkey', 'INTEGER', references='nation(n_nationkey)'),
        PostgresColumn('s_phone', 'CHAR(15)'),
        PostgresColumn('s_acctbal', 'DECIMAL(15,2)'),
        PostgresColumn('s_comment', 'VARCHAR(101)'),
    ]

    partsupp = [
        PostgresColumn('ps_partkey', 'INTEGER', primary_key=True, references='part(p_partkey)'),
        PostgresColumn('ps_suppkey', 'INTEGER', primary_key=True, references='supplier(s_suppkey)'),
        PostgresColumn('ps_availqty', 'INTEGER'),
        PostgresColumn('ps_supplycost', 'DECIMAL(15,2)'),
        PostgresColumn('ps_comment', 'VARCHAR(255)'),
    ]

    lineitem = [
        PostgresColumn('l_orderkey', 'INTEGER', primary_key=True, references='orders(o_orderkey)'),
        PostgresColumn('l_partkey', 'INTEGER', references='part(p_partkey)'),
        PostgresColumn('l_suppkey', 'INTEGER', references='supplier(s_suppkey)'),
        PostgresColumn('l_linenumber', 'INTEGER', primary_key=True),
        PostgresColumn('l_quantity', 'DECIMAL(15,2)'),
        PostgresColumn('l_extendedprice', 'DECIMAL(15,2)'),
        PostgresColumn('l_discount', 'DECIMAL(15,2)'),
        PostgresColumn('l_tax', 'DECIMAL(15,2)'),
        PostgresColumn('l_returnflag', 'CHAR(1)'),
        PostgresColumn('l_linestatus', 'CHAR(1)'),
        PostgresColumn('l_shipdate', 'DATE'),
        PostgresColumn('l_commitdate', 'DATE'),
        PostgresColumn('l_receiptdate', 'DATE'),
        PostgresColumn('l_shipinstruct', 'CHAR(25)'),
        PostgresColumn('l_shipmode', 'CHAR(10)'),
        PostgresColumn('l_comment', 'VARCHAR(44)'),
    ]

    # Custom tables (not part of the original TPC-H schema)

    knows = [
        PostgresColumn('k_custkey1', 'INTEGER', primary_key=True, references='customer(c_custkey)'),
        PostgresColumn('k_custkey2', 'INTEGER', primary_key=True, references='customer(c_custkey)'),
        PostgresColumn('k_startdate', 'DATE', primary_key=True),
        PostgresColumn('k_source', 'VARCHAR(20)'), # e.g. 'organic', 'social_media', etc.
        PostgresColumn('k_comment', 'TEXT'), # optional free-form comment about the relationship
        PostgresColumn('k_strength', 'DOUBLE PRECISION'), # Strength of the relationship (0.00 to 1.00)
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
