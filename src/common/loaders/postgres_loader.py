from abc import ABC, abstractmethod
import os
import time
from common.drivers import PostgresDriver
from common.config import DatasetName
from common.utils import time_quantity

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

class PostgresLoader(ABC):

    """A class to load data into a Postgres database."""
    def __init__(self, driver: PostgresDriver):
        self._driver = driver

    @abstractmethod
    def dataset(self) -> DatasetName:
        """Returns the name of the loader (for display purposes)."""
        pass

    @abstractmethod
    def _get_schemas(self) -> dict[str, list[ColumnSchema]]:
        """Returns the schemas for each entity kind. The order of kinds is important for creating tables with foreign key dependencies."""
        pass

    @abstractmethod
    def _get_indexes(self) -> list[IndexSchema]:
        """Returns the list of indexes to be created after tables are created."""
        pass

    def run(self, import_directory: str, do_reset: bool):
        title = f'--- {self.dataset().label()} Postgres Loader ---'
        print(title)
        print(f'Connecting to Postgres at: {self._driver.config.host}:{self._driver.config.port}')
        print('-' * len(title) + '\n')

        self.__times = dict[str, float]()
        self._import_directory = import_directory
        self.__check_files()

        if do_reset:
            print('Resetting database...')
            self.__reset_database()
            print('Database reset completed.')

        print('\nCreating schema...')
        for entity, schema in self._get_schemas().items():
            self.__create_kind_schema(entity, schema)
        for index in self._get_indexes():
            self.__create_index(index)
        print('Schema created.')

        print('\nLoading data...')
        for entity, schema in self._get_schemas().items():
            self.__populate_kind(entity, schema)
        print('Data loading completed.')

        return self.__times

    def __check_files(self):
        """Verify that all files exist in the import directory."""
        for kind in self._get_schemas().keys():
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

    def __create_kind_schema(self, entity: str, schema: list[ColumnSchema]):
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

        with self._driver.cursor() as cursor:
            cursor.execute(query)
            print(f'Created table "{entity}".')

    def __create_index(self, index: IndexSchema):
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

        with self._driver.cursor() as cursor:
            cursor.execute(query)

    def __populate_kind(self, entity: str, schema: list[ColumnSchema]):
        print(f'Loading table "{entity}"...')

        query = f'''
            COPY {escape(entity)}
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
        with open(path, "r") as file:
            with self._driver.cursor() as cursor:
                cursor.copy_expert(query, file)
        self.__times[entity] = time_quantity.to_base(time.perf_counter() - start, 's')

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

            cursor.execute(f'CREATE DATABASE {escape(database_name)} WITH OWNER = {escape(driver.config.user)}')
            if cursor.rowcount > 0:
                print(f'Created database "{database_name}".')
    finally:
        driver.put_connection(connection)

    return True

def escape(kind: str) -> str:
    """Escapes SQL identifiers to prevent clash with reserved keywords."""
    return f'"{kind}"'
