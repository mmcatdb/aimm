from abc import ABC, abstractmethod
import os
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

    def run(self, import_directory: str, do_reset: bool):
        title = f'--- {self.name()} Neo4j Loader ---'
        print(title)
        print(f'Connecting to Neo4j at: {self._driver.config.host}:{self._driver.port}')
        print('-' * len(title) + '\n')

        try:
            self._driver.verify()
            print('Successfully connected to Neo4j.')
        except Exception as e:
            raise Exception(f'Failed to connect to Neo4j: {e}')

        self._check_files(import_directory)

        if do_reset:
            print('Resetting database...')
            self._dao.reset_database()
            print('Database reset completed.')

        print('Creating constraints...')
        for constraint in self._define_constraints():
            self._dao.execute(constraint)
        print('Constraints created.')

        print('Loading data...')
        self._load_data()
        print('Data loading completed.')

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
