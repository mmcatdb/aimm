from typing_extensions import override
from common.drivers import Neo4jDriver, cypher
from common.daos.base_dao import BaseDAO

class Neo4jDAO(BaseDAO):
    def __init__(self, driver: Neo4jDriver):
        self.driver = driver

    @override
    def find(self, entity_name, query_params) -> list[dict]:
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

        query = f'MATCH (n:{entity_name}) WHERE {where_clause} RETURN n'

        results = self._execute_query(query, params)
        return [res['n'] for res in results]

    @override
    def insert(self, entity_name, data):
        query = f'CREATE (n:{entity_name} $props)'
        self._execute_query(query, {'props': data})

    @override
    def create_schema(self, entity_name, schema):
        # Could create constraints in place of a schema, but it's not necessary (at least for now)
        pass

    @override
    def delete_all_from(self, entity_name):
        query = f'MATCH (n:{entity_name}) DETACH DELETE n'
        self._execute_query(query)
        print(f'All data from "{entity_name}" has been deleted in Neo4j.')

    @override
    def drop_entity(self, entity_name):
        self.delete_all_from(entity_name)
        print(f'All nodes with label "{entity_name}" have been dropped in Neo4j.')

    def _execute_query(self, query: str, params=None):
        with self.driver.session() as session:
            result = session.run(cypher(query), params)
            return [record.data() for record in result]
