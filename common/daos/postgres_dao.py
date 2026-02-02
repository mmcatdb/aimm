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
    def find(self, entity: str, query_params) -> list[dict[Any, Any]]:
        query = f'SELECT * FROM {entity} WHERE '
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
    def insert(self, entity: str, data: dict):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f'INSERT INTO {entity} ({columns}) VALUES ({placeholders})'

        with self.driver.cursor() as cursor:
            cursor.execute(query, list(data.values()))

    @override
    def create_kind_schema(self, entity: str, schema: list[dict]):
        columns_def = [f'{col["name"]} {col["type"].replace("PRIMARY KEY", "").strip()}' for col in schema]

        # Identify primary key columns from the primary_key flag
        pk_cols = [col['name'] for col in schema if col.get('primary_key')]
        assert pk_cols is not None, f'No primary key defined for entity "{entity}". Please specify a primary key(s) in the schema.'

        pk_def = f', PRIMARY KEY ({", ".join(pk_cols)})'

        # Build and execute the final query
        query = f'CREATE TABLE IF NOT EXISTS {entity} ({", ".join(columns_def)}{pk_def})'

        with self.driver.cursor() as cursor:
            cursor.execute(query)
            print(f'Table "{entity}" created or already exists in PostgreSQL.')

    @override
    def drop_kinds(self, populate_order: list[str]) -> None:
        for entity in reversed(populate_order):
            try:
                with self.driver.cursor() as cursor:
                    cursor.execute(f'DROP TABLE IF EXISTS {entity}')
                    print(f'Table "{entity}" has been dropped in PostgreSQL.')
            except Exception as e:
                print(f'Skipping delete for {entity}: {e}')

    @override
    def reset_database(self) -> None:
        query = """
        DO $$
        DECLARE
            s RECORD;
        BEGIN
            FOR s IN
                SELECT nspname
                FROM pg_namespace
                WHERE nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
                AND nspname NOT LIKE 'pg_temp_%'
            LOOP
                EXECUTE format('DROP SCHEMA %I CASCADE', s.nspname);
            END LOOP;

            EXECUTE 'CREATE SCHEMA public';
            EXECUTE 'GRANT ALL ON SCHEMA public TO public';
        END $$;
        """
        with self.driver.cursor() as cursor:
            cursor.execute(query)
            print('PostgreSQL database has been reset.')

    def execute(self, query: str, params=None):
        with self.driver.cursor(cursor_factory = RealDictCursor) as cursor:
            cursor.execute(query, params)
            if cursor.description:
                results = cursor.fetchall()
                return results

        raise ValueError('Query did not return any results.')
