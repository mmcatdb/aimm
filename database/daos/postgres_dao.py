import psycopg2

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
            conditions.append(f"{key} = %s")
            params.append(value)
        
        query += " AND ".join(conditions)
        
        cursor.execute(query, tuple(params))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        cursor.close()
        return results
    

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