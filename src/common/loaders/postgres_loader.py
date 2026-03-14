import csv
from abc import ABC, abstractmethod
import os
from common.daos.postgres_dao import PostgresDAO, ColumnSchema, IndexSchema
from common.drivers import PostgresDriver

class PostgresLoader(ABC):
    """A class to load data into a Postgres database."""
    def __init__(self, driver: PostgresDriver):
        self._driver = driver
        self._dao = PostgresDAO(driver)

    @abstractmethod
    def name(self) -> str:
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
        title = f'--- {self.name()} Postgres Loader ---'
        print(title)
        print(f'Connecting to Postgres at: {self._driver.config.host}:{self._driver.config.port}')
        print('-' * len(title) + '\n')

        self._import_directory = import_directory
        self._check_files()

        if do_reset:
            print('Resetting database...')
            self._dao.reset_database()
            print('Database reset completed.')

        print('\nCreating schema...')
        for entity, schema in self._get_schemas().items():
            self._dao.create_kind_schema(entity, schema)
        for index in self._get_indexes():
            self._dao.create_index(index)
        print('Schema created.')

        print('\nLoading data...')
        for entity, schema in self._get_schemas().items():
            self._populate_table(entity, schema)
        print('Data loading completed.')

    def _check_files(self):
        """Verify that all files exist in the import directory."""
        for kind in self._get_schemas().keys():
            filename = kind + '.tbl'
            filepath = os.path.join(self._import_directory, filename)
            if not os.path.isfile(filepath):
                raise Exception(f'Required file not found in import directory: {filepath}')

    def _populate_table(self, entity: str, schema: list[ColumnSchema]):
        print(f'Loading table "{entity}"...')

        filename = entity + '.tbl'
        path = os.path.join(self._import_directory, filename)

        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                data = {}
                for i, column in enumerate(schema):
                    data[column.name] = row[i]

                self._dao.insert(entity, data)
