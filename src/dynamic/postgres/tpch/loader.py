from typing_extensions import override

from core.loaders.postgres_loader import PostgresColumn, PostgresIndex, PostgresLoader


class TpchPostgresLoader(PostgresLoader):
    @override
    def _get_kinds(self) -> dict[str, list[PostgresColumn]]:
        c = PostgresColumn
        return {
            'region': [
                c('r_regionkey', 'INTEGER', primary_key=True),
                c('r_name', 'CHAR(25)'),
                c('r_comment', 'VARCHAR(152)'),
            ],
            'nation': [
                c('n_nationkey', 'INTEGER', primary_key=True),
                c('n_name', 'CHAR(25)'),
                c('n_regionkey', 'INTEGER', references='region(r_regionkey)'),
                c('n_comment', 'VARCHAR(152)'),
            ],
            'customer': [
                c('c_custkey', 'INTEGER', primary_key=True),
                c('c_name', 'VARCHAR(25)'),
                c('c_address', 'VARCHAR(40)'),
                c('c_nationkey', 'INTEGER', references='nation(n_nationkey)'),
                c('c_phone', 'CHAR(15)'),
                c('c_acctbal', 'DECIMAL(15,2)'),
                c('c_mktsegment', 'CHAR(10)'),
                c('c_comment', 'VARCHAR(117)'),
            ],
            'orders': [
                c('o_orderkey', 'INTEGER', primary_key=True),
                c('o_custkey', 'INTEGER', references='customer(c_custkey)'),
                c('o_orderstatus', 'CHAR(1)'),
                c('o_totalprice', 'DECIMAL(15,2)'),
                c('o_orderdate', 'DATE'),
                c('o_orderpriority', 'CHAR(15)'),
                c('o_clerk', 'CHAR(15)'),
                c('o_shippriority', 'INTEGER'),
                c('o_comment', 'VARCHAR(79)'),
            ],
            'part': [
                c('p_partkey', 'INTEGER', primary_key=True),
                c('p_name', 'VARCHAR(55)'),
                c('p_mfgr', 'CHAR(25)'),
                c('p_brand', 'CHAR(10)'),
                c('p_type', 'VARCHAR(25)'),
                c('p_size', 'INTEGER'),
                c('p_container', 'CHAR(10)'),
                c('p_retailprice', 'DECIMAL(15,2)'),
                c('p_comment', 'VARCHAR(23)'),
            ],
            'supplier': [
                c('s_suppkey', 'INTEGER', primary_key=True),
                c('s_name', 'CHAR(25)'),
                c('s_address', 'VARCHAR(40)'),
                c('s_nationkey', 'INTEGER', references='nation(n_nationkey)'),
                c('s_phone', 'CHAR(15)'),
                c('s_acctbal', 'DECIMAL(15,2)'),
                c('s_comment', 'VARCHAR(101)'),
            ],
            'partsupp': [
                c('ps_partkey', 'INTEGER', primary_key=True, references='part(p_partkey)'),
                c('ps_suppkey', 'INTEGER', primary_key=True, references='supplier(s_suppkey)'),
                c('ps_availqty', 'INTEGER'),
                c('ps_supplycost', 'DECIMAL(15,2)'),
                c('ps_comment', 'VARCHAR(255)'),
            ],
            'lineitem': [
                c('l_orderkey', 'INTEGER', primary_key=True, references='orders(o_orderkey)'),
                c('l_partkey', 'INTEGER', references='part(p_partkey)'),
                c('l_suppkey', 'INTEGER', references='supplier(s_suppkey)'),
                c('l_linenumber', 'INTEGER', primary_key=True),
                c('l_quantity', 'DECIMAL(15,2)'),
                c('l_extendedprice', 'DECIMAL(15,2)'),
                c('l_discount', 'DECIMAL(15,2)'),
                c('l_tax', 'DECIMAL(15,2)'),
                c('l_returnflag', 'CHAR(1)'),
                c('l_linestatus', 'CHAR(1)'),
                c('l_shipdate', 'DATE'),
                c('l_commitdate', 'DATE'),
                c('l_receiptdate', 'DATE'),
                c('l_shipinstruct', 'CHAR(25)'),
                c('l_shipmode', 'CHAR(10)'),
                c('l_comment', 'VARCHAR(44)'),
            ],
            'knows': [
                c('k_custkey1', 'INTEGER', primary_key=True, references='customer(c_custkey)'),
                c('k_custkey2', 'INTEGER', primary_key=True, references='customer(c_custkey)'),
                c('k_startdate', 'DATE'),
                c('k_source', 'VARCHAR(20)'),
                c('k_comment', 'VARCHAR(117)'),
                c('k_strength', 'DOUBLE PRECISION'),
            ],
        }

    @override
    def _get_constraints(self) -> list[PostgresIndex]:
        return [
            PostgresIndex('customer', ['c_nationkey']),
            PostgresIndex('orders', ['o_custkey']),
            PostgresIndex('orders', ['o_orderdate']),
            PostgresIndex('lineitem', ['l_orderkey']),
            PostgresIndex('lineitem', ['l_partkey', 'l_suppkey']),
            PostgresIndex('lineitem', ['l_shipdate']),
            PostgresIndex('partsupp', ['ps_suppkey']),
            PostgresIndex('knows', ['k_custkey2']),
        ]


def export():
    return TpchPostgresLoader()
