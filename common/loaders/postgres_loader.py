import csv
from abc import ABC, abstractmethod
import argparse
import os
from common.config import Config
from common.daos.postgres_dao import PostgresDAO
from common.drivers import PostgresDriver

class PostgresLoader(ABC):
    """A class to load data into a Postgres database."""
    def __init__(self, config: Config, driver: PostgresDriver):
        self._config = config
        self._driver = driver
        self._dao = PostgresDAO(driver)

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the loader (for display purposes)."""
        pass

    @abstractmethod
    def _get_schemas(self) -> dict[str, list[dict]]:
        """Returns the schemas for each entity kind. The order of kinds is important for creating tables with foreign key dependencies."""
        pass

    @abstractmethod
    def _get_indexes(self) -> list[dict]:
        """Returns the list of indexes to be created after tables are created."""
        pass

    def run(self):
        args = self._parse_args()

        title = f'--- {self.name()} Postgres Loader ---'
        print(title)
        print(f'Reset database: {args.reset_database}')
        print(f'Connecting to Postgres at: {self._driver.config.host}:{self._driver.config.port}')
        print('-' * len(title) + '\n')

        import_directory = args.data_dir or self._config.import_directory

        try:
            self._check_files(import_directory)

            if args.reset_database:
                print('Resetting database...')
                self._dao.reset_database()
                print('Database reset complete.')

            print('Creating schema...')
            schemas = self._get_schemas()
            for entity, schema in schemas.items():
                self._dao.create_kind_schema(entity, schema)
            for index in self._get_indexes():
                self._dao.create_index(index)
            print('Schema created.')

            print('Loading data...')
            for entity, schema in schemas.items():
                self._populate_table(entity, schema)
            print('Data loading complete.')
        except Exception as e:
            print(f'\nError: {e}')
        finally:
            self._driver.close()

        print('\nScript finished successfully.')

    def _parse_args(self):
        parser = argparse.ArgumentParser(description=f'Load {self.name()} data into a Postgres database.')
        parser.add_argument(
            '--data-dir',
            type=str,
            default=None,
            help=f'Path to the directory containing the {self.name()} .tbl files. If not specified, reads from "IMPORT_DIRECTORY" in .env.'
        )
        parser.add_argument(
            '--reset-database',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Set to --no-reset-database to skip clearing the database beforehand.'
        )

        return parser.parse_args()

    def _check_files(self, import_directory: str):
        """Verify files exist in the import directory."""
        for kind in self._get_schemas().keys():
            filename = kind + '.tbl'
            filepath = os.path.join(import_directory, filename)
            if not os.path.isfile(filepath):
                raise Exception(f'Required file not found in import directory: {filepath}')

    def _populate_table(self, entity: str, schema: list[dict]):
        filename = entity + '.tbl'
        path = os.path.join(self._config.import_directory, filename)

        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                # Skip empty or malformed rows
                if not row or all(col == '' for col in row):
                    continue

                data = {}
                for i, column in enumerate(schema):
                    if i < len(row) and column['name']:
                        data[column['name']] = row[i]

                # Drop potential empty key from final delimiter
                data = {k: v for k, v in data.items() if k}
                if data:
                    self._dao.insert(entity, data)

        print(f'Loaded data into table "{entity}".')

