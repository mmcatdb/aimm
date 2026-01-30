from typing import Any
import psycopg2
from psycopg2.extras import RealDictCursor

from .base_dao import BaseDAO
from common.databases import Postgres

class PostgresDAO(BaseDAO):
    def __init__(self, postgres: Postgres):
        self.postgres = postgres

    def _execute_query(self, query, params=None):
        with self.postgres.cursor(cursor_factory = RealDictCursor) as cursor:
            cursor.execute(query, params)
            if cursor.description:
                results = cursor.fetchall()
                return results

        raise ValueError('Query did not return any results.')

    def find(self, entity_name, query_params) -> list[dict[Any, Any]]:
        query = f'SELECT * FROM {entity_name} WHERE '
        conditions = []
        params = []
        for key, value in query_params.items():
            if key.endswith('__in'):
                conditions.append(f'{key[:-4]} IN %s')
                params.append(tuple(value))
            else:
                conditions.append(f'{key} = %s')
                params.append(value)

        query += ' AND '.join(conditions)

        with self.postgres.cursor() as cursor:
            cursor.execute(query, tuple(params))
            if cursor.description is None:
                return []

            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return results

    def get_all_lineitems(self):
        return self._execute_query('SELECT * FROM lineitem;')

    def get_orders_by_daterange(self, start_date, end_date):
        return self._execute_query('SELECT * FROM orders WHERE o_orderdate BETWEEN %s AND %s;', (start_date, end_date))

    def get_all_customers(self):
        return self._execute_query('SELECT * FROM customer;')

    def get_orders_by_keyrange(self, start_key, end_key):
        return self._execute_query('SELECT * FROM orders WHERE o_orderkey BETWEEN %s AND %s;', (start_key, end_key))

    def count_orders_by_month(self):
        return self._execute_query("""
            SELECT COUNT(o_orderkey) AS order_count,
                   TO_CHAR(o_orderdate, 'YYYY-MM') AS order_month
            FROM orders
            GROUP BY order_month;
        """)

    def get_max_price_by_ship_month(self):
        return self._execute_query("""
            SELECT TO_CHAR(l_shipdate, 'YYYY-MM') AS ship_month,
                   MAX(l_extendedprice) AS max_price
            FROM lineitem
            GROUP BY ship_month;
        """)

    # --- Part / Supplier / PartSupp ---
    def get_all_parts(self):
        return self._execute_query('SELECT * FROM part;')

    def get_parts_by_size_range(self, min_size, max_size):
        return self._execute_query('SELECT * FROM part WHERE p_size BETWEEN %s AND %s;', (min_size, max_size))

    def get_all_suppliers(self):
        return self._execute_query('SELECT * FROM supplier;')

    def get_suppliers_by_nation(self, nation_key):
        return self._execute_query('SELECT * FROM supplier WHERE s_nationkey = %s;', (nation_key,))

    def get_partsupp_for_part(self, partkey):
        return self._execute_query('SELECT * FROM partsupp WHERE ps_partkey = %s;', (partkey,))

    def get_lowest_cost_supplier_for_part(self, partkey):
        res = self._execute_query("""
            SELECT ps.*, s.s_name, s.s_acctbal
            FROM partsupp ps
            JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
            WHERE ps.ps_partkey = %s
            ORDER BY ps.ps_supplycost ASC
            LIMIT 1;
        """, (partkey,))
        return res[0] if res else None

    def count_suppliers_per_part(self):
        return self._execute_query("""
            SELECT ps_partkey AS partkey, COUNT(*) AS supplier_count
            FROM partsupp
            GROUP BY ps_partkey
            ORDER BY ps_partkey;
        """)

    def avg_supplycost_by_part_size(self):
        return self._execute_query("""
            SELECT p.p_size, AVG(ps.ps_supplycost) AS avg_supplycost
            FROM part p
            JOIN partsupp ps ON p.p_partkey = ps.ps_partkey
            GROUP BY p.p_size
            ORDER BY p.p_size;
        """)

    def insert(self, entity_name, data):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f'INSERT INTO {entity_name} ({columns}) VALUES ({placeholders})'

        with self.postgres.cursor() as cursor:
            cursor.execute(query, list(data.values()))

    def create_schema(self, entity_name, schema):
        columns_def = [f'{col["name"]} {col["type"].replace("PRIMARY KEY", "").strip()}' for col in schema]

        # Identify primary key columns from the primary_key flag
        pk_cols = [col['name'] for col in schema if col.get('primary_key')]
        if not pk_cols:
            raise ValueError(f'No primary key defined for entity "{entity_name}". Please specify a primary key(s) in the schema.')

        pk_def = f', PRIMARY KEY ({", ".join(pk_cols)})'

        # Build and execute the final query
        query = f'CREATE TABLE IF NOT EXISTS {entity_name} ({", ".join(columns_def)}{pk_def})'

        with self.postgres.cursor() as cursor:
            cursor.execute(query)
            print(f'Table "{entity_name}" created or already exists in PostgreSQL.')

    def delete_all_from(self, entity_name):
        connection = self.postgres.get_connection()
        try:
            with connection.cursor() as cursor:
                query = f'TRUNCATE TABLE {entity_name} RESTART IDENTITY'
                cursor.execute(query)
            connection.commit()
            print(f'All data from "{entity_name}" has been deleted in PostgreSQL.')
        except psycopg2.errors.UndefinedTable:
            connection.rollback()
            print(f'Table "{entity_name}" does not exist, skipping TRUNCATE.')
        finally:
            self.postgres.put_connection(connection)

    def drop_entity(self, entity_name):
        with self.postgres.cursor() as cursor:
            query = f'DROP TABLE IF EXISTS {entity_name}'
            cursor.execute(query)
            print(f'Table "{entity_name}" has been dropped in PostgreSQL.')
