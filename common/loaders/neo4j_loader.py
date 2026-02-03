from abc import ABC, abstractmethod
import argparse
import os
import shutil
from common.config import Config
from common.daos.neo4j_dao import Neo4jDAO
from common.drivers import Neo4jDriver

class Neo4jLoader(ABC):
    """A class to load data into a Neo4j database."""
    def __init__(self, config: Config, driver: Neo4jDriver):
        self._config = config
        self._driver = driver
        self._dao = Neo4jDAO(driver)

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the loader (for display purposes)."""
        pass

    @abstractmethod
    def _get_kinds(self) -> list[str]:
        """Returns the list of entity kinds to be loaded. Used for file management."""
        pass

    @abstractmethod
    def _define_constraints(self) -> list[str]:
        """Creates unique constraints for primary keys to speed up data loading."""
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """Loads data from all .tbl files into Neo4j. The order of loading is important to ensure relationships can be formed."""
        pass

    def run(self):
        args = self._parse_args()

        title = f'--- {self.name()} Neo4j Loader ---'
        print(title)
        print(f'Reset database: {args.reset_database}')
        print(f'Connecting to Neo4j at: {self._driver.config.host}:{self._driver.config.port}')
        print('-' * len(title) + '\n')

        self._check_connection()

        import_directory = args.import_dir or self._config.import_directory
        data_directory = args.data_dir

        copied_files = None

        try:
            if data_directory:
                copied_files = self._copy_files(import_directory, data_directory)
            else:
                self._check_files(import_directory)

            if args.reset_database:
                print('Resetting database...')
                self._dao.reset_database()
                print('Database reset complete.')

            print('Creating constraints...')
            for constraint in self._define_constraints():
                self._dao.execute(constraint)
            print('Constraints created.')

            print('Loading data...')
            self._load_data()
            print('Data loading complete.')
        except Exception as e:
            print(f'\nError: {e}')
        finally:
            self._driver.close()

            if copied_files:
                self._clean_copied_files(copied_files)

        print('\nScript finished successfully.')

    def _parse_args(self):
        parser = argparse.ArgumentParser(description=f'Load {self.name()} data into a Neo4j database.')
        parser.add_argument(
            '--data-dir',
            type=str,
            default=None,
            help=f'Path to the directory containing the {self.name()} .tbl files. Files will be copied from there to the Neo4j import directory. If not specified, the files are expected to be already present in the Neo4j import directory.'
        )
        parser.add_argument(
            '--import-dir',
            type=str,
            default=None,
            help=(
                'Path to Neo4j\'s import directory. If not specified, reads from "IMPORT_DIRECTORY" in .env.\n'
                'Common locations:\n'
                '  - Linux (Debian/RPM): /var/lib/neo4j/import\n'
                '  - macOS (Homebrew):   /usr/local/var/neo4j/import or /opt/homebrew/var/neo4j/import\n'
                '  - Docker (compose):   Leave empty (check the compose file)\n'
                '  - Docker (custom):    Mapped volume (often /import inside container)\n'
                '  - Neo4j Desktop:      Open App -> Manage -> Open Folder -> Import'
            )
        )
        parser.add_argument(
            '--reset-database',
            action=argparse.BooleanOptionalAction,
            default=True,
            help='Set to --no-reset-database to skip clearing the database beforehand.'
        )

        return parser.parse_args()

    def _check_connection(self):
        try:
            self._driver.verify()
            print('Successfully connected to Neo4j.')
        except Exception as e:
            raise Exception(f'Failed to connect to Neo4j: {e}')

    def _copy_files(self, import_directory: str, data_directory: str) -> list[str]:
        """Copy files from data_directory to the import directory. Returns list of copied file paths for cleanup."""
        copied_files = []
        if not os.path.isdir(data_directory):
            raise Exception(f'Data directory does not exist: {data_directory}')

        print(f'Copying .tbl files from "{data_directory}" to "{import_directory}"...')
        for kind in self._get_kinds():
            filename = kind + '.tbl'
            src = os.path.join(data_directory, filename)
            dst = os.path.join(import_directory, filename)
            if not os.path.isfile(src):
                raise Exception(f'Required file not found: {src}')

            shutil.copy2(src, dst)
            copied_files.append(dst)
            print(f'  Copied: {filename}')

        return copied_files

    def _clean_copied_files(self, copied_files: list[str]):
        if copied_files:
            print('\nCleaning up copied .tbl files from the import directory...')
            for filepath in copied_files:
                try:
                    os.remove(filepath)
                    print(f'  Removed: {os.path.basename(filepath)}')
                except OSError as e:
                    print(f'  Warning: Could not remove {filepath}: {e}')

    def _check_files(self, import_directory: str):
        """Verify files exist in the import directory."""
        print(f'Using .tbl files directly from the import directory: "{import_directory}"')
        for kind in self._get_kinds():
            filename = kind + '.tbl'
            filepath = os.path.join(import_directory, filename)
            if not os.path.isfile(filepath):
                raise Exception(f'Required file not found in import directory: {filepath}')

    def _load_csv(self, entity: str, content: str, note: str | None = None):
        if note:
            print(f'Loading {entity} and {note}...')
        else:
            print(f'Loading {entity}...')

        self._dao.execute(f'''
        LOAD CSV FROM 'file:///{entity}.tbl' AS row FIELDTERMINATOR '|'
        CALL (row) {{ {content} }} IN TRANSACTIONS OF 500 ROWS
        ''')
