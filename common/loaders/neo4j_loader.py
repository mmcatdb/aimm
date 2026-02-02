from abc import ABC, abstractmethod
import argparse
import os
import shutil
from common.config import Config
from common.drivers import Neo4jDriver, cypher

class Neo4jLoader(ABC):
    """A class to load data into a Neo4j database."""
    def __init__(self, config: Config, neo4j: Neo4jDriver):
        self._config = config
        self._neo4j = neo4j

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the loader (for display purposes)."""
        pass

    @abstractmethod
    def _get_kinds(self) -> list[str]:
        """Returns the list of entity kinds to be loaded. Used for file management."""
        pass

    @abstractmethod
    def _create_constraints(self) -> None:
        """Creates unique constraints for primary keys to speed up data loading."""
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """Loads data from all .tbl files into Neo4j. The order of loading is important to ensure relationships can be formed."""
        pass

    def run(self):
        args = self._parse_args()

        print(f'--- {self.name()} Neo4j Loader ---')
        print(f'Reset database: {args.reset_database}')
        print(f'Connecting to Neo4j at: {self._neo4j.config.host}:{self._neo4j.config.port}')
        print('----------------------------\n')

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
                self._reset_database()

            print('Creating constraints...')
            self._create_constraints()
            print('Constraints created.')

            print('\n--- Starting Data Loading ---')
            self._load_data()
            print('\n--- Data Loading Complete ---')

            print('\nScript finished successfully.')
        except Exception as e:
            print(f'\nError: {e}')
        finally:
            self._neo4j.close()

            if copied_files:
                self._clean_copied_files(copied_files)

    #region App logic

    def _parse_args(self):
        parser = argparse.ArgumentParser(description=f'Load {self.name()} data into a Neo4j database.')
        parser.add_argument(
            '--config',
            type=str,
            default=None,
            help=f'Path to config file (default: {Config.DEFAULT_CONFIG_PATH})'
        )
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
            self._neo4j.verify()
            print('Successfully connected to Neo4j.')
        except Exception as e:
            raise Exception(f'Failed to connect to Neo4j: {e}')

    def _copy_files(self, import_directory: str, data_directory: str) -> list[str]:
        """ Copy files from data_directory to neo4j import directory. Returns list of copied file paths for cleanup."""
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
            print('\nCleaning up copied .tbl files from import directory...')
            for filepath in copied_files:
                try:
                    os.remove(filepath)
                    print(f'  Removed: {os.path.basename(filepath)}')
                except OSError as e:
                    print(f'  Warning: Could not remove {filepath}: {e}')

    def _check_files(self, import_directory: str):
        # Verify files exist in neo4j import directory
        print(f'Using .tbl files directly from Neo4j import directory: "{import_directory}"')
        for kind in self._get_kinds():
            filename = kind + '.tbl'
            filepath = os.path.join(import_directory, filename)
            if not os.path.isfile(filepath):
                raise Exception(f'Required file not found in import directory: {filepath}')

    #endregion App logic
    #region DB Utils

    def _run_query(self, query: str, parameters=None):
        """
        Executes a Cypher query that doesn't need to return data.
        Ensures the result is fully consumed within the session.
        """
        with self._neo4j.session() as session:
            result = session.run(cypher(query), parameters or {})
            result.consume()

    def _run_scalar(self, query: str, parameters=None, key=None):
        """
        Executes a Cypher query expected to return a single record.
        Returns the value for 'key' or the first value in the record.
        """
        with self._neo4j.session() as session:
            rec = session.run(cypher(query), parameters or {}).single()
            if rec is None:
                return None
            if key is None:
                values = list(rec.values())
                return values[0] if values else None
            return rec.get(key)

    def _reset_database(self):
        """
        Drops all constraints and deletes all nodes and relationships in batches.
        """
        print('Resetting database...')

        existing_constraints = self.__get_constraint_names()
        constraints = [f'DROP CONSTRAINT {name} IF EXISTS' for name in existing_constraints]
        for constraint in constraints:
            try:
                self._run_query(constraint)
            except Exception as e:
                print(f'Constraint not found or could not be dropped: {constraint}... Error: {e}')

        print('Deleting all nodes and relationships in batches...')
        batch_size = 10_000
        total_deleted = 0

        delete_batch_query = """
        MATCH (n)
        WITH n LIMIT $limit
        WITH collect(n) AS nodes
        WITH nodes, size(nodes) AS deleted
        FOREACH (x IN nodes | DETACH DELETE x)
        RETURN deleted
        """

        while True:
            deleted = self._run_scalar(delete_batch_query, parameters={'limit': batch_size}, key='deleted') or 0
            total_deleted += deleted
            print(f'Deleted batch: {deleted} nodes; total deleted so far: {total_deleted}')
            if deleted == 0:
                break

        # Verify emptiness and print out counts
        remaining_nodes = self._run_scalar('MATCH (n) RETURN count(n) AS nodes', key='nodes') or 0
        remaining_rels = self._run_scalar('MATCH ()-[r]-() RETURN count(r) AS rels', key='rels') or 0

        if remaining_nodes == 0 and remaining_rels == 0:
            print(f'Database has been cleared. Nodes: {remaining_nodes}, Relationships: {remaining_rels}')
        else:
            print(f'Warning: Database not empty after reset. Nodes: {remaining_nodes}, Relationships: {remaining_rels}')

    def __get_constraint_names(self):
        query = 'SHOW CONSTRAINTS YIELD name'
        with self._neo4j.session() as session:
            result = session.run(query)
            return [record['name'] for record in result]

    def _create_node(self, entity: str, content: str, note: str | None = None):
        if note:
            print(f'Loading {entity} and {note}...')
        else:
            print(f'Loading {entity}...')

        prefix = f'LOAD CSV FROM \'file:///{entity}.tbl\' AS row FIELDTERMINATOR \'|\'\n'
        query = prefix + content
        self._run_query(query)

    #endregion DB Utils
