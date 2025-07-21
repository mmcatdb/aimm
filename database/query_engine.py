import yaml
# from daos.mongo_dao import MongoDAO
# from daos.neo4j_dao import Neo4jDAO
from daos.postgres_dao import PostgresDAO


class QueryEngine:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.schema_mapping = self.config['schema_mapping']
        self.daos = {
            'postgres': PostgresDAO(self.config['postgres']),
            # 'mongodb': MongoDAO(self.config['mongodb']),
            # 'neo4j': Neo4jDAO(self.config['neo4j'])
        }

    def get_dao_for_entity(self, entity_name):
        db_type = self.schema_mapping.get(entity_name)
        if not db_type:
            raise ValueError(f"Entity '{entity_name}' not found in schema mapping.")
        return self.daos[db_type]


    def find(self, entity_name, query_params):
        dao = self.get_dao_for_entity(entity_name)
        return dao.find(entity_name, query_params)


    def find_lineitems_for_customer(self, customer_name):
        print(f"Searching for lineitems for customer: {customer_name}")

        # Find customer
        customer_dao = self.get_dao_for_entity('customer')
        customers = customer_dao.find('customer', {'c_name': customer_name})
        if not customers:
            return []
        customer = customers[0]
        print(f"Found customer: {customer}")

        # Find orders for the customer
        orders_dao = self.get_dao_for_entity('orders')
        orders = orders_dao.find('orders', {'o_custkey': customer['c_custkey']})
        if not orders:
            return []
        print(f"Found {len(orders)} orders for the customer.")


        # Find lineitems for the orders
        lineitem_dao = self.get_dao_for_entity('lineitem')
        all_lineitems = []
        for order in orders:
            lineitems = lineitem_dao.find('lineitem', {'l_orderkey': order['o_orderkey']})
            all_lineitems.extend(lineitems)
        
        print(f"Found {len(all_lineitems)} total lineitems.")
        return all_lineitems


    def disconnect_all(self):
        for dao in self.daos.values():
            dao.disconnect()
