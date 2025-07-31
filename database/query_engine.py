import yaml
from daos.mongo_dao import MongoDAO
from daos.neo4j_dao import Neo4jDAO
from daos.postgres_dao import PostgresDAO
import time

class QueryEngine:
    def __init__(self, config_path='config.yaml', schema_mapping=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.schema_mapping = schema_mapping or self.config['schema_mapping']
        print(self.schema_mapping)
        
        self.daos = {
            'postgres': PostgresDAO(self.config['postgres']),
            'mongodb': MongoDAO(self.config['mongodb']),
            'neo4j': Neo4jDAO(self.config['neo4j'])
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
        # Find the customer 
        customer_dao = self.get_dao_for_entity('customer')
        customers = customer_dao.find('customer', {'c_name': customer_name})
        if not customers: return []
        customer = customers[0]

        # Find all orders for the customer
        orders_dao = self.get_dao_for_entity('orders')
        orders = orders_dao.find('orders', {'o_custkey': customer['c_custkey']})
        if not orders: return []

        # Get all order keys from the results
        order_keys = [order['o_orderkey'] for order in orders]

        # Find all lineitems for the given orders
        lineitem_dao = self.get_dao_for_entity('lineitem')
        all_lineitems = lineitem_dao.find('lineitem', {'l_orderkey__in': order_keys})
        
        return all_lineitems


    def run_queries(self, customer_name="Customer#000000007", verbose=True):
        # Queries from: https://github.com/wsawa-q/evaluation-of-db-performance/blob/main/evaluation/database/mysql/queries.md
        
        start = checkpoint = time.time()
        
        if verbose:
            print(f"Customers stored in: {self.schema_mapping.get('customer')}")
            print(f"Orders stored in: {self.schema_mapping.get('orders')}")
            print(f"Lineitems stored in: {self.schema_mapping.get('lineitem')}")
            print("=" * 50)

        lineitem_dao = self.get_dao_for_entity('lineitem')
        customer_dao = self.get_dao_for_entity('customer')
        order_dao = self.get_dao_for_entity('orders')
        
        
        
        # -----------  A1  ----------- 
        lineitems = lineitem_dao.get_all_lineitems()
        if verbose:
            print(f"Found {len(lineitems)} lineitems.")
            print(f"Time taken for A1: {time.time() - start:.2f} seconds")
            print("-" * 50)
        checkpoint = time.time()

        # -----------  A2  ----------- 
        orders = order_dao.get_orders_by_daterange('1996-01-01', '1996-12-31')
        if verbose:
            print(f"Found {len(orders)} orders.")
            print(f"Time taken for A2: {time.time() - checkpoint:.2f} seconds")
            print("-" * 50)
        checkpoint = time.time()
        
        
        # -----------  A3  ----------- 
        customers = customer_dao.get_all_customers()
        if verbose:
            print(f"Found {len(customers)} customers.")
            print(f"Time taken for A3: {time.time() - checkpoint:.2f} seconds")
            print("-" * 50)
        checkpoint = time.time()


        # -----------  A4  ----------- 
        orders = order_dao.get_orders_by_keyrange("1000", "50000")
        if verbose:
            print(f"Found {len(orders)} orders.")
            print(f"Time taken for A4: {time.time() - checkpoint:.2f} seconds")
            print("-" * 50)
        checkpoint = time.time()
        
        
        # -----------  B1  ----------- 
        orders_by_month = order_dao.count_orders_by_month()
        if verbose:
            print(f"Found orders by month. Total number of months: {len(orders_by_month)}")
            print(f"Time taken for B1: {time.time() - checkpoint:.2f} seconds")
            print("-" * 50)
        checkpoint = time.time()

        # -----------  B2  ----------- 
        max_price = lineitem_dao.get_max_price_by_ship_month()
        if verbose:
            print(f"Found max price by ship month. Total number of months: {len(max_price)}")
            print(f"Time taken for B2: {time.time() - checkpoint:.2f} seconds")
            print("-" * 50)
        checkpoint = time.time()

        # -----------  Join  ---------
        lineitems = self.find_lineitems_for_customer(customer_name)
        if verbose:
            print(f"Found {len(lineitems)} lineitems for customer: {customer_name}")
            print(f"Time taken for Join: {time.time() - checkpoint:.2f} seconds")
            print("-" * 50)

        if verbose:
            print(f"Finished. Total time taken: {time.time() - start:.2f} seconds")
        
        return round(time.time() - start, 2)



    def disconnect_all(self):
        for dao in self.daos.values():
            dao.disconnect()
