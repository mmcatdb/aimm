from typing_extensions import override
from common.drivers import Neo4jDriver, cypher
from common.daos.base_dao import BaseDAO

class Neo4jDAO(BaseDAO):
    def __init__(self, driver: Neo4jDriver):
        self.driver = driver

    @override
    def find(self, entity: str, query_params) -> list[dict]:
        conditions = []
        params = {}
        for key, value in query_params.items():
            prop = key.split('__')[0]
            if key.endswith('__in'):
                conditions.append(f'n.{prop} IN ${prop}')
                params[prop] = value
            else:
                conditions.append(f'n.{prop} = ${prop}')
                params[prop] = value


        where_clause = ' AND '.join(conditions)

        query = f'MATCH (n:{entity}) WHERE {where_clause} RETURN n'

        results = self.execute(query, params)
        return [res['n'] for res in results]

    @override
    def insert(self, entity: str, data: dict):
        query = f'CREATE (n:{entity} $props)'
        self.execute(query, {'props': data})

    @override
    def drop_kinds(self, populate_order: list[str]) -> None:
        for entity in reversed(populate_order):
            # This is not ideal but what can we do. At least this forces strict naming conventions.
            if entity.isupper():
                query = f'MATCH ()-[r:{entity}]-() DELETE r'
            else:
                query = f'MATCH (n:{entity}) DETACH DELETE n'

            try:
                self.execute(query)
                print(f'All "{entity}" nodes have been dropped in Neo4j.')
            except Exception as e:
                print(f'Skipping delete for {entity}: {e}')

    @override
    def reset_database(self) -> None:
        existing_constraints = self._get_constraint_names()
        constraints = [f'DROP CONSTRAINT {name} IF EXISTS' for name in existing_constraints]
        for constraint in constraints:
            try:
                self.execute(constraint)
            except Exception as e:
                print(f'Constraint not found or could not be dropped: {constraint}... Error: {e}')

        print('Deleting all nodes and relationships in batches...')
        batch_size = 10_000
        total_deleted = 0

        delete_batch_query = '''
        MATCH (n)
        WITH n LIMIT $limit
        WITH collect(n) AS nodes
        WITH nodes, size(nodes) AS deleted
        FOREACH (x IN nodes | DETACH DELETE x)
        RETURN deleted
        '''

        while True:
            deleted = self.execute_to_scalar(delete_batch_query, parameters={'limit': batch_size}, key='deleted') or 0
            total_deleted += deleted
            print(f'Deleted batch: {deleted} nodes, total deleted so far: {total_deleted}')
            if deleted == 0:
                break

        # Verify emptiness and print out counts
        remaining_nodes = self.execute_to_scalar('MATCH (n) RETURN count(n) AS nodes', key='nodes') or 0
        remaining_rels = self.execute_to_scalar('MATCH ()-[r]-() RETURN count(r) AS rels', key='rels') or 0

        if remaining_nodes + remaining_rels > 0:
            print(f'Warning: Database not empty after reset. Nodes: {remaining_nodes}, Relationships: {remaining_rels}')

    def _get_constraint_names(self):
        query = 'SHOW CONSTRAINTS YIELD name'
        with self.driver.session() as session:
            result = session.run(query)
            return [record['name'] for record in result]

    def execute(self, query: str, params=None):
        with self.driver.session() as session:
            result = session.run(cypher(query), params)
            return [record.data() for record in result]

    def execute_to_scalar(self, query: str, parameters=None, key=None):
        """
        Executes a Cypher query expected to return a single record.
        Returns the value for 'key' or the first value in the record.
        """
        with self.driver.session() as session:
            rec = session.run(cypher(query), parameters or {}).single()
            if rec is None:
                return None
            if key is None:
                values = list(rec.values())
                return values[0] if values else None
            return rec.get(key)
