from abc import abstractmethod
import os
import time
from typing_extensions import override
from core.drivers import PostgresDriver
from core.files import open_input
from core.query import SchemaId
from core.utils import time_quantity
from .base_loader import BaseLoader

class PostgresColumn:
    def __init__(self, name: str, type: str, primary_key = False, references: str | None = None):
        self.name = name
        self.type = type
        self.primary_key = primary_key
        self.references = references

class PostgresIndex:
    def __init__(self, kind: str, columns: list[str], is_unique=False, where=None):
        self.kind = kind
        self.columns = columns
        self.is_unique = is_unique
        self.where = where

class PostgresLoader(BaseLoader):
    """A class to load data into a Postgres database."""

    _driver: PostgresDriver

    @abstractmethod
    def _get_kinds(self) -> dict[str, list[PostgresColumn]]:
        """Returns the schema for each entity kind. The order matters."""
        pass

    @abstractmethod
    def _get_constraints(self) -> list[PostgresIndex]:
        """Returns the list of constraints to be created."""
        pass

    @override
    def run(self, driver: PostgresDriver, schema_id: SchemaId, import_directory: str, do_reset: bool):
        self._reset(driver, schema_id, import_directory)

        print(f'Loading data to Postgres at: {self._driver.config.host}:{self._driver.config.port}')

        self.__check_files()

        if do_reset:
            print('Resetting database...')
            self.__reset_database()
            print('Database reset completed.')

        print('\nCreating schema...')
        for entity, columns in self._get_kinds().items():
            self.__create_kind(entity, columns)
        for index in self._get_constraints():
            self.__create_index(index)
        print('Schema created.')

        print('\nLoading data...')
        for entity, columns in self._get_kinds().items():
            self.__populate_kind(entity, columns)
        print('Data loading completed.')

        return self._times

    def __check_files(self):
        """Verify that all files exist in the import directory."""
        for kind in self._get_kinds().keys():
            path = os.path.join(self._import_directory, kind + '.tbl')
            if not os.path.isfile(path):
                raise Exception(f'Required file not found in import directory: {path}')

    def __reset_database(self):
        query = '''
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
        '''
        with self._driver.cursor() as cursor:
            cursor.execute(query)

    def __create_kind(self, name: str, columns: list[PostgresColumn]):
        column_defs = []
        for col in columns:
            definition = f'{_escape(col.name)} {col.type}'
            if col.references:
                ref_table, ref_column = col.references.split('(')
                ref_column = ref_column.rstrip(')')
                definition += f' REFERENCES {_escape(ref_table)}({_escape(ref_column)})'

            column_defs.append(definition)

        # Identify primary key columns from the primary_key flag
        escaped_columns = [_escape(col.name) for col in columns if col.primary_key]
        assert escaped_columns, f'No primary key defined for entity "{name}". Please specify a primary key(s) in the schema.'

        pk_def = f', PRIMARY KEY ({", ".join(escaped_columns)})'

        # Build and execute the final query
        query = f'CREATE TABLE IF NOT EXISTS {_escape(name)} ({", ".join(column_defs)}{pk_def})'

        with self._driver.cursor() as cursor:
            cursor.execute(query)
            print(f'Created table "{name}".')

    def __create_index(self, index: PostgresIndex):
        """
        CREATE UNIQUE INDEX uniq_reviews_product_user ON reviews(product_id, user_id)
        {}'CREATE INDEX idx_products_active ON products(is_active) WHERE is_active = TRUE',
        """
        table = index.kind
        escaped_columns = ', '.join([_escape(col) for col in index.columns])
        unique = 'UNIQUE ' if index.is_unique else ''
        where_clause = f' WHERE {index.where}' if index.where else ''
        prefix = 'uniq' if index.is_unique else 'idx'
        index_name = f'{prefix}_{"_".join([table] + index.columns)}'

        query = f'CREATE {unique}INDEX IF NOT EXISTS {_escape(index_name)} ON {_escape(table)} ({escaped_columns}){where_clause}'

        with self._driver.cursor() as cursor:
            cursor.execute(query)

    def __populate_kind(self, entity: str, columns: list[PostgresColumn]):
        print(f'Loading table "{entity}"...')

        query = f'''
            COPY {_escape(entity)}
            FROM STDIN
            WITH (
                FORMAT CSV,
                DELIMITER '|',
                NULL '',
                HEADER FALSE
            )
        '''

        path = os.path.join(self._import_directory, entity + '.tbl')

        start = time.perf_counter()
        with open_input(path) as file:
            with self._driver.cursor() as cursor:
                cursor.copy_expert(query, file)
        self._times[entity] = time_quantity.to_base(time.perf_counter() - start, 's')

def create_database_if_not_exists(driver: PostgresDriver, database_name: str) -> bool:
    """Returns True if the database was created, False if it already existed."""
    connection = driver.get_connection()
    # Create database can't be run inside a transation so this is required.
    connection.autocommit = True

    try:
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1 FROM pg_database WHERE datname = %s', (database_name, ))
            exists = cursor.fetchone() is not None
            if exists:
                print(f'Database "{database_name}" already exists.')
                return False

            cursor.execute(f'CREATE DATABASE {_escape(database_name)} WITH OWNER = {_escape(driver.config.user)}')
            if cursor.rowcount > 0:
                print(f'Created database "{database_name}".')
    finally:
        driver.put_connection(connection)

    return True

def _escape(kind: str) -> str:
    """Escapes SQL identifiers to prevent clash with reserved keywords."""
    return f'"{kind}"'
