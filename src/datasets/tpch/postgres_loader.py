from typing_extensions import override
from common.loaders.postgres_loader import PostgresLoader
from common.daos.postgres_dao import ColumnSchema, IndexSchema

class TpchPostgresLoader(PostgresLoader):
    @override
    def name(self) -> str:
        return 'TPC-H'

    # TODO Add not null constraints.

    @override
    def _get_schemas(self) -> dict[str, list[ColumnSchema]]:
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

        return {
            'region': region,
            'nation': nation,
            'customer': customer,
            'orders': orders,
            'part': part,
            'supplier': supplier,
            'partsupp': partsupp,
            'lineitem': lineitem,
        }

    @override
    def _get_indexes(self) -> list[IndexSchema]:
        return [
            # TODO add more indexes based on the queries we want to run
        ]
