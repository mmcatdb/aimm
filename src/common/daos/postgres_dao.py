from typing import Any
from typing_extensions import override
from psycopg2.extras import RealDictCursor
from common.daos.base_dao import BaseDAO
from common.drivers import PostgresDriver

class ColumnSchema:
    def __init__(self, name: str, type: str, primary_key = False, references: str | None = None):
        self.name = name
        self.type = type
        self.primary_key = primary_key
        self.references = references

class IndexSchema:
    def __init__(self, kind: str, columns: list[str], is_unique=False, where=None):
        self.kind = kind
        self.columns = columns
        self.is_unique = is_unique
        self.where = where

class PostgresDAO(BaseDAO):
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
    def insert(self, entity: str, data: dict):
        columns = ', '.join([escape(col) for col in data.keys()])
        placeholders = ', '.join(['%s'] * len(data))
        query = f'INSERT INTO {escape(entity)} ({columns}) VALUES ({placeholders})'

        with self.driver.cursor() as cursor:
            cursor.execute(query, list(data.values()))

    def create_kind_schema(self, entity: str, schema: list[ColumnSchema]):
        column_defs = []
        for col in schema:
            definition = f'{escape(col.name)} {col.type}'
            if col.references:
                ref_table, ref_column = col.references.split('(')
                ref_column = ref_column.rstrip(')')
                definition += f' REFERENCES {escape(ref_table)}({escape(ref_column)})'

            column_defs.append(definition)

        # Identify primary key columns from the primary_key flag
        pk_cols = [escape(col.name) for col in schema if col.primary_key]
        assert pk_cols is not None, f'No primary key defined for entity "{entity}". Please specify a primary key(s) in the schema.'

        pk_def = f', PRIMARY KEY ({", ".join(pk_cols)})'

        # Build and execute the final query
        query = f'CREATE TABLE IF NOT EXISTS {escape(entity)} ({", ".join(column_defs)}{pk_def})'

        with self.driver.cursor() as cursor:
            cursor.execute(query)
            print(f'Created table "{entity}".')

    def create_index(self, index: IndexSchema):
        """
        CREATE UNIQUE INDEX uniq_reviews_product_user ON reviews(product_id, user_id)
        {}'CREATE INDEX idx_products_active ON products(is_active) WHERE is_active = TRUE',
        """
        table = index.kind
        columns = ', '.join([escape(col) for col in index.columns])
        unique = 'UNIQUE ' if index.is_unique else ''
        where_clause = f' WHERE {index.where}' if index.where else ''
        prefix = 'uniq' if index.is_unique else 'idx'
        index_name = f'{prefix}_{"_".join([table] + index.columns)}'

        query = f'CREATE {unique}INDEX IF NOT EXISTS {escape(index_name)} ON {escape(table)} ({columns}){where_clause}'

        with self.driver.cursor() as cursor:
            cursor.execute(query)

    @override
    def drop_kinds(self, populate_order: list[str]):
        for entity in reversed(populate_order):
            try:
                with self.driver.cursor() as cursor:
                    cursor.execute(f'DROP TABLE IF EXISTS {escape(entity)}')
                    print(f'Dropped table "{entity}".')
            except Exception as e:
                print(f'Skipping delete for {entity}: {e}')

    def create_database_if_not_exists(self, database_name: str) -> bool:
        """Returns True if the database was created, False if it already existed."""
        connection = self.driver.get_connection()
        # Create database can't be run inside a transation so this is required.
        connection.autocommit = True

        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT 1 FROM pg_database WHERE datname = %s', (database_name, ))
                exists = cursor.fetchone() is not None
                if exists:
                    print(f'Database "{database_name}" already exists.')
                    return False

                cursor.execute(f'CREATE DATABASE {escape(database_name)} WITH OWNER = {escape(self.driver.config.user)}')
                if cursor.rowcount > 0:
                    print(f'Created database "{database_name}".')
        finally:
            self.driver.put_connection(connection)

        return True

    @override
    def reset_database(self):
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

    def execute(self, query: str, params=None):
        with self.driver.cursor(cursor_factory = RealDictCursor) as cursor:
            cursor.execute(query, params)
            if cursor.description:
                results = cursor.fetchall()
                return results

        raise ValueError('Query did not return any results.')

def escape(kind: str) -> str:
    """Escapes SQL identifiers to prevent clash with reserved keywords."""
    return f'"{kind}"'
