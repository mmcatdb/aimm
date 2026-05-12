from abc import abstractmethod
import os
import time
from core.drivers import Neo4jDriver, cypher
from core.query import SchemaId
from core.utils import ProgressTracker, print_warning, time_quantity
from .base_loader import BaseLoader

class Neo4jLoader(BaseLoader):
    """A class to load data into a Neo4j database."""

    _driver: Neo4jDriver

    @abstractmethod
    def _get_kinds(self) -> list[str]:
        """Returns the schema for each entity kind. The order matters."""
        pass

    @abstractmethod
    def _get_constraints(self) -> list[str]:
        """Returns the list of constraints to be created."""
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """Loads data from all .tbl files into Neo4j. The order of loading is important to ensure relationships can be formed."""
        pass

    def run(self, driver: Neo4jDriver, schema_id: SchemaId, import_directory: str, do_reset: bool):
        self._reset(driver, schema_id, import_directory)

        print(f'Loading data to Neo4j at: {self._driver.config.host}:{self._driver.port}')

        try:
            self._driver.verify()
            print('Successfully connected to Neo4j.')
        except Exception as e:
            print_warning('Failed to connect to Neo4j.', e)
            raise e

        self.__check_files()

        if do_reset:
            print('\nResetting database...')
            self.__reset_database()
            print('Database reset completed.')

        print('\nCreating constraints...')
        for constraint in self._get_constraints():
            self._driver.execute(constraint)
        print('Constraints created.')

        print('\nLoading data...')
        self._load_data()
        print('Data loading completed.')

        return self._times

    def __check_files(self):
        """Verify that all files exist in the import directory."""
        print(f'Using .tbl files directly from the import directory: "{self._import_directory}"')
        for kind in self._get_kinds():
            path = os.path.join(self._import_directory, kind + '.tbl')
            if not os.path.isfile(path):
                raise Exception(f'Required file not found in import directory: {path}')

    def __reset_database(self):
        existing_constraints = self.__get_constraint_names()
        constraints = [f'DROP CONSTRAINT {name} IF EXISTS' for name in existing_constraints]
        for constraint in constraints:
            try:
                self._driver.execute(constraint)
            except Exception as e:
                print_warning(f'Constraint not found or could not be dropped: {constraint}', e)

        total_nodes: int | None = None
        try:
            # Count all nodes and relationships before deletion for progress tracking
            total_nodes = self.__execute_to_scalar('MATCH (n) RETURN count(n) AS count', key='count') or 0
        except Exception as e:
            print_warning('Could not count nodes before deletion.', e)

        progress = ProgressTracker.limited(total_nodes) if total_nodes is not None else ProgressTracker.unlimited()
        progress.start('Deleting existing data... ')

        batch_size = 10_000

        delete_batch_query = '''
            MATCH (n)
            WITH n LIMIT $limit
            WITH collect(n) AS nodes
            WITH nodes, size(nodes) AS deleted
            FOREACH (x IN nodes | DETACH DELETE x)
            RETURN deleted
        '''

        while True:
            deleted = self.__execute_to_scalar(delete_batch_query, parameters={'limit': batch_size}, key='deleted') or 0
            progress.track(deleted)
            if deleted == 0:
                break

        progress.finish()

        # Verify emptiness and print out counts
        remaining_nodes = self.__execute_to_scalar('MATCH (n) RETURN count(n) AS nodes', key='nodes') or 0
        remaining_rels = self.__execute_to_scalar('MATCH ()-[r]-() RETURN count(r) AS rels', key='rels') or 0

        if remaining_nodes + remaining_rels > 0:
            print(f'Warning: Database not empty after reset. Nodes: {remaining_nodes}, Relationships: {remaining_rels}')

    def __get_constraint_names(self):
        query = 'SHOW CONSTRAINTS YIELD name'
        with self._driver.session() as session:
            result = session.run(query)
            return [record['name'] for record in result]

    def _load_csv(self, entity: str, csv_name: str, content: str, note: str | None = None):
        if note:
            print(f'Loading {entity} and {note}...')
        else:
            print(f'Loading {entity}...')

        start = time.perf_counter()
        self._driver.execute(f'''
            LOAD CSV FROM 'file:///{self._schema_id}/{csv_name}.tbl' AS row FIELDTERMINATOR '|'
            CALL (row) {{ {content} }} IN TRANSACTIONS OF 500 ROWS
        ''')
        self._times[entity] = time_quantity.to_base(time.perf_counter() - start, 's')

    def _create_relationship(self, label: str, from_entity: str, from_key: str, to_entity: str, to_key: str):
        print(f'Creating {from_entity} -> {to_entity} relationships...')

        start = time.perf_counter()
        self._driver.execute(f'''
            MATCH (from:{from_entity}), (to:{to_entity} {{{to_key}: from.{from_key}}})
            CALL (from, to) {{
                CREATE (from)-[:{label}]->(to)
            }} IN TRANSACTIONS OF 1000 ROWS
        ''')
        self._times[label] = time_quantity.to_base(time.perf_counter() - start, 's')

    def __execute_to_scalar(self, query: str, parameters=None, key=None):
        """
        Executes a Cypher query expected to return a single record.
        Returns the value for 'key' or the first value in the record.
        """
        with self._driver.session() as session:
            rec = session.run(cypher(query), parameters or {}).single()
            if rec is None:
                return None
            if key is None:
                values = list(rec.values())
                return values[0] if values else None
            return rec.get(key)
