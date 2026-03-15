from typing import Any
from typing_extensions import override
from common.drivers import PostgresDriver
from datasets.tpch.daos.tpch_dao import TpchDAO

class TpchPostgresDAO(TpchDAO):
    def __init__(self, driver: PostgresDriver):
        self.driver = driver

    @override
    def find(self, entity: str, query_params) -> list[dict[Any, Any]]:
        query = f'SELECT * FROM {escape(entity)} WHERE '
        conditions = []
        params = []
        for key, value in query_params.items():
            if key.endswith('__in'):
                conditions.append(f'{escape(key[:-4])} IN %s')
                params.append(tuple(value))
            else:
                conditions.append(f'{escape(key)} = %s')
                params.append(value)

        query += ' AND '.join(conditions)

        with self.driver.cursor() as cursor:
            cursor.execute(query, tuple(params))
            if cursor.description is None:
                return []

            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return results

    @override
    def get_all_lineitems(self):
        return self.driver.execute('SELECT * FROM lineitem')

    @override
    def get_orders_by_daterange(self, start_date, end_date):
        return self.driver.execute('SELECT * FROM orders WHERE o_orderdate BETWEEN %s AND %s', (start_date, end_date))

    @override
    def get_all_customers(self):
        return self.driver.execute('SELECT * FROM customer')

    @override
    def get_orders_by_keyrange(self, start_key, end_key):
        return self.driver.execute('SELECT * FROM orders WHERE o_orderkey BETWEEN %s AND %s', (start_key, end_key))

    @override
    def count_orders_by_month(self):
        return self.driver.execute("""
            SELECT COUNT(o_orderkey) AS order_count,
                   TO_CHAR(o_orderdate, 'YYYY-MM') AS order_month
            FROM orders
            GROUP BY order_month
        """)

    @override
    def get_max_price_by_ship_month(self):
        return self.driver.execute("""
            SELECT TO_CHAR(l_shipdate, 'YYYY-MM') AS ship_month,
                   MAX(l_extendedprice) AS max_price
            FROM lineitem
            GROUP BY ship_month
        """)

    # --- Part / Supplier / PartSupp ---
    @override
    def get_all_parts(self):
        return self.driver.execute('SELECT * FROM part')

    @override
    def get_parts_by_size_range(self, min_size, max_size):
        return self.driver.execute('SELECT * FROM part WHERE p_size BETWEEN %s AND %s', (min_size, max_size))

    @override
    def get_all_suppliers(self):
        return self.driver.execute('SELECT * FROM supplier')

    @override
    def get_suppliers_by_nation(self, nation_key):
        return self.driver.execute('SELECT * FROM supplier WHERE s_nationkey = %s', (nation_key,))

    @override
    def get_partsupp_for_part(self, partkey):
        return self.driver.execute('SELECT * FROM partsupp WHERE ps_partkey = %s', (partkey,))

    @override
    def get_lowest_cost_supplier_for_part(self, partkey):
        res = self.driver.execute("""
            SELECT ps.*, s.s_name, s.s_acctbal
            FROM partsupp ps
            JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
            WHERE ps.ps_partkey = %s
            ORDER BY ps.ps_supplycost ASC
            LIMIT 1
        """, (partkey,))
        return res[0] if res else None

    @override
    def count_suppliers_per_part(self):
        return self.driver.execute("""
            SELECT ps_partkey AS partkey, COUNT(*) AS supplier_count
            FROM partsupp
            GROUP BY ps_partkey
            ORDER BY ps_partkey
        """)

    @override
    def avg_supplycost_by_part_size(self):
        return self.driver.execute("""
            SELECT p.p_size, AVG(ps.ps_supplycost) AS avg_supplycost
            FROM part p
            JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
            GROUP BY p.p_size
            ORDER BY p.p_size
        """)

def escape(kind: str) -> str:
    """Escapes SQL identifiers to prevent clash with reserved keywords."""
    return f'"{kind}"'
