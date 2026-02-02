from typing import Any
from typing_extensions import override
from psycopg2.errors import UndefinedTable
from psycopg2.extras import RealDictCursor
from common.daos.base_dao import BaseDAO
from common.drivers import PostgresDriver

class PostgresDAO(BaseDAO):
    def __init__(self, driver: PostgresDriver):
        self.driver = driver

    @override
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

        with self.driver.cursor() as cursor:
            cursor.execute(query, tuple(params))
            if cursor.description is None:
                return []

            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return results

    @override
    def insert(self, entity_name, data):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f'INSERT INTO {entity_name} ({columns}) VALUES ({placeholders})'

        with self.driver.cursor() as cursor:
            cursor.execute(query, list(data.values()))

    @override
    def create_schema(self, entity_name, schema):
        columns_def = [f'{col["name"]} {col["type"].replace("PRIMARY KEY", "").strip()}' for col in schema]

        # Identify primary key columns from the primary_key flag
        pk_cols = [col['name'] for col in schema if col.get('primary_key')]
        assert pk_cols is not None, f'No primary key defined for entity "{entity_name}". Please specify a primary key(s) in the schema.'

        pk_def = f', PRIMARY KEY ({", ".join(pk_cols)})'

        # Build and execute the final query
        query = f'CREATE TABLE IF NOT EXISTS {entity_name} ({", ".join(columns_def)}{pk_def})'

        with self.driver.cursor() as cursor:
            cursor.execute(query)
            print(f'Table "{entity_name}" created or already exists in PostgreSQL.')

    @override
    def delete_all_from(self, entity_name):
        connection = self.driver.get_connection()
        try:
            with connection.cursor() as cursor:
                query = f'TRUNCATE TABLE {entity_name} RESTART IDENTITY'
                cursor.execute(query)
            connection.commit()
            print(f'All data from "{entity_name}" has been deleted in PostgreSQL.')
        except UndefinedTable:
            connection.rollback()
            print(f'Table "{entity_name}" does not exist, skipping TRUNCATE.')
        finally:
            self.driver.put_connection(connection)

    @override
    def drop_entity(self, entity_name):
        with self.driver.cursor() as cursor:
            query = f'DROP TABLE IF EXISTS {entity_name}'
            cursor.execute(query)
            print(f'Table "{entity_name}" has been dropped in PostgreSQL.')

    def _execute_query(self, query, params=None):
        with self.driver.cursor(cursor_factory = RealDictCursor) as cursor:
            cursor.execute(query, params)
            if cursor.description:
                results = cursor.fetchall()
                return results

        raise ValueError('Query did not return any results.')
