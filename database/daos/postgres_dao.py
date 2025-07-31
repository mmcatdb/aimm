import psycopg2
from psycopg2.extras import RealDictCursor

from .base_dao import BaseDAO


class PostgresDAO(BaseDAO):
    def __init__(self, config):
        self.config = config
        self.conn = psycopg2.connect(**self.config) if config else None

    def connect(self):
        self.conn = psycopg2.connect(**self.config)

    def disconnect(self):
        if self.conn:
            self.conn.close()

    def find(self, entity_name, query_params):
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        
        query = f"SELECT * FROM {entity_name} WHERE "
        conditions = []
        params = []
        for key, value in query_params.items():
            if key.endswith("__in"):
                conditions.append(f"{key[:-4]} IN %s")
                params.append(tuple(value))
            else:
                conditions.append(f"{key} = %s")
                params.append(value)

        query += " AND ".join(conditions)
        
        cursor.execute(query, tuple(params))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        cursor.close()
        return results

    def _execute_query(self, query, params=None):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if cur.description:
                results = cur.fetchall()
                return results

            # In case of non-select statements return None
            return None


    def get_all_lineitems(self):
        return self._execute_query("SELECT * FROM lineitem;")

    def get_orders_by_daterange(self, start_date, end_date):
        query = "SELECT * FROM orders WHERE o_orderdate BETWEEN %s AND %s;"
        return self._execute_query(query, (start_date, end_date))

    def get_all_customers(self):
        return self._execute_query("SELECT * FROM customer;")

    def get_orders_by_keyrange(self, start_key, end_key):
        query = "SELECT * FROM orders WHERE o_orderkey BETWEEN %s AND %s;"
        return self._execute_query(query, (start_key, end_key))

    def count_orders_by_month(self):
        query = """
            SELECT COUNT(o_orderkey) AS order_count, 
                   TO_CHAR(o_orderdate, 'YYYY-MM') AS order_month
            FROM orders
            GROUP BY order_month;
        """
        return self._execute_query(query)

    def get_max_price_by_ship_month(self):
        query = """
            SELECT TO_CHAR(l_shipdate, 'YYYY-MM') AS ship_month,
                   MAX(l_extendedprice) AS max_price
            FROM lineitem
            GROUP BY ship_month;
        """
        return self._execute_query(query)

    def insert(self, entity_name, data):
        with self.conn.cursor() as cur:
            # Prepare the insert statement
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['%s'] * len(data))
            query = f"INSERT INTO {entity_name} ({columns}) VALUES ({placeholders})"
            cur.execute(query, list(data.values()))
            self.conn.commit()

    def create_schema(self, entity_name, schema):
        with self.conn.cursor() as cur:
            columns_def = [f"{col['name']} {col['type'].replace('PRIMARY KEY', '').strip()}" for col in schema]
            
            # Identify primary key columns from the primary_key flag
            pk_cols = [col['name'] for col in schema if col.get('primary_key')]
            if not pk_cols:
                raise ValueError(f"No primary key defined for entity '{entity_name}'. Please specify a primary key(s) in the schema.")
            
            pk_def = f", PRIMARY KEY ({', '.join(pk_cols)})"

            # Build and execute the final query
            query = f"CREATE TABLE IF NOT EXISTS {entity_name} ({', '.join(columns_def)}{pk_def})"
            cur.execute(query)
            self.conn.commit()
            print(f"Table '{entity_name}' created or already exists in PostgreSQL.")


    def delete_all_from(self, entity_name):
        with self.conn.cursor() as cur:
            query = f"TRUNCATE TABLE {entity_name} RESTART IDENTITY"
            cur.execute(query)
            self.conn.commit()
            print(f"All data from '{entity_name}' has been deleted in PostgreSQL.")


    def drop_entity(self, entity_name):
        with self.conn.cursor() as cur:
            query = f"DROP TABLE IF EXISTS {entity_name}"
            cur.execute(query)
            self.conn.commit()
            print(f"Table '{entity_name}' has been dropped in PostgreSQL.")