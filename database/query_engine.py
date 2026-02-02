import time
from common.driver_provider import DriverProvider
from common.drivers import MongoDriver, Neo4jDriver, PostgresDriver
from datasets.tpch.postgres_dao import TpchPostgresDAO
from datasets.tpch.mongo_dao import TpchMongoDAO
from datasets.tpch.neo4j_dao import TpchNeo4jDAO

class QueryEngine:
    def __init__(self, dbs: DriverProvider, schema_mapping: dict[str, str]):
        self.schema_mapping = schema_mapping
        self.daos = {
            'postgres': TpchPostgresDAO(dbs.get_typed('postgres', PostgresDriver)),
            'mongo': TpchMongoDAO(dbs.get_typed('mongo', MongoDriver)),
            'neo4j': TpchNeo4jDAO(dbs.get_typed('neo4j', Neo4jDriver)),
        }

    def get_dao_for_entity(self, entity_name):
        db_type = self.schema_mapping[entity_name]
        if not db_type:
            raise ValueError(f'Entity "{entity_name}" not found in schema mapping.')
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

    def run_queries(self, customer_name='Customer#000000007', verbose=True) -> float:
        # Queries from: https://github.com/wsawa-q/evaluation-of-db-performance/blob/main/evaluation/database/mysql/queries.md

        start = checkpoint = time.time()

        if verbose:
            print(f'Customers stored in: {self.schema_mapping["customer"]}')
            print(f'Orders stored in: {self.schema_mapping["orders"]}')
            print(f'Lineitems stored in: {self.schema_mapping["lineitem"]}')
            print(f'Parts stored in: {self.schema_mapping["part"]}')
            print(f'Suppliers stored in: {self.schema_mapping["supplier"]}')
            print(f'PartSupp stored in: {self.schema_mapping["partsupp"]}')
            print('=' * 50)

        lineitem_dao = self.get_dao_for_entity('lineitem')
        customer_dao = self.get_dao_for_entity('customer')
        order_dao = self.get_dao_for_entity('orders')

        # -----------  A1  -----------
        lineitems = lineitem_dao.get_all_lineitems()
        if verbose:
            print(f'Found {len(lineitems)} lineitems.')
            print(f'Time taken for A1: {time.time() - start:.2f} seconds')
            print('-' * 50)
        checkpoint = time.time()

        # -----------  A2  -----------
        orders = order_dao.get_orders_by_daterange('1996-01-01', '1996-12-31')
        if verbose:
            print(f'Found {len(orders)} orders.')
            print(f'Time taken for A2: {time.time() - checkpoint:.2f} seconds')
            print('-' * 50)
        checkpoint = time.time()

        # -----------  A3  -----------
        customers = customer_dao.get_all_customers()
        if verbose:
            print(f'Found {len(customers)} customers.')
            print(f'Time taken for A3: {time.time() - checkpoint:.2f} seconds')
            print('-' * 50)
        checkpoint = time.time()

        # -----------  A4  -----------
        orders = order_dao.get_orders_by_keyrange('1000', '50000')
        if verbose:
            print(f'Found {len(orders)} orders.')
            print(f'Time taken for A4: {time.time() - checkpoint:.2f} seconds')
            print('-' * 50)
        checkpoint = time.time()

        # -----------  B1  -----------
        orders_by_month = order_dao.count_orders_by_month()
        if verbose:
            print(f'Found orders by month. Total number of months: {len(orders_by_month)}')
            print(f'Time taken for B1: {time.time() - checkpoint:.2f} seconds')
            print('-' * 50)
        checkpoint = time.time()

        # -----------  B2  -----------
        max_price = lineitem_dao.get_max_price_by_ship_month()
        if verbose:
            print(f'Found max price by ship month. Total number of months: {len(max_price)}')
            print(f'Time taken for B2: {time.time() - checkpoint:.2f} seconds')
            print('-' * 50)
        checkpoint = time.time()

        # -----------  Join  ---------
        lineitems = self.find_lineitems_for_customer(customer_name)
        if verbose:
            print(f'Found {len(lineitems)} lineitems for customer: {customer_name}')
            print(f'Time taken for Join: {time.time() - checkpoint:.2f} seconds')
            print('-' * 50)
        checkpoint = time.time()

        self.run_part_supplier_queries(verbose=verbose)

        if verbose:
            print(f'Finished. Total time taken: {time.time() - start:.2f} seconds')

        return round(time.time() - start, 2)

    def run_part_supplier_queries(self, verbose=True):
        part_dao = self.get_dao_for_entity('part')
        supplier_dao = self.get_dao_for_entity('supplier')
        partsupp_dao = self.get_dao_for_entity('partsupp')
        start = checkpoint = time.time()

        # P1
        parts = part_dao.get_all_parts()
        if verbose:
            print(f'P1) Parts count: {len(parts)} (all parts)')
            print(f'Time P1: {time.time() - checkpoint:.2f}s')
        checkpoint = time.time()

        # P2
        parts_size_range = part_dao.get_parts_by_size_range(1, 20)
        if verbose:
            print(f'P2) Parts size 1-20: {len(parts_size_range)}')
            print(f'Time P2: {time.time() - checkpoint:.2f}s')
        checkpoint = time.time()

        # S1
        suppliers = supplier_dao.get_all_suppliers()
        if verbose:
            print(f'S1) Suppliers count: {len(suppliers)}')
            print(f'Time S1: {time.time() - checkpoint:.2f}s')
        checkpoint = time.time()

        # S2
        suppliers_nation = supplier_dao.get_suppliers_by_nation('1')
        if verbose:
            print(f'S2) Suppliers nation=1: {len(suppliers_nation)}')
            print(f'Time S2: {time.time() - checkpoint:.2f}s')
        checkpoint = time.time()

        # PS1
        partkey_example = 1
        ps_for_part = partsupp_dao.get_partsupp_for_part(partkey_example)
        if verbose:
            print(f'PS1) PartSupp rows for part {partkey_example}: {len(ps_for_part)}')
            print(f'Time PS1: {time.time() - checkpoint:.2f}s')
        checkpoint = time.time()

        # PS2
        lowest_cost = partsupp_dao.get_lowest_cost_supplier_for_part(partkey_example)
        if verbose:
            print(f'PS2) Lowest cost supplier for part {partkey_example}: {"found" if lowest_cost else "none"}')
            print(f'Time PS2: {time.time() - checkpoint:.2f}s')
        checkpoint = time.time()

        # AGG1
        suppliers_per_part = partsupp_dao.count_suppliers_per_part()
        if verbose:
            print(f'AGG1) Suppliers per part rows: {len(suppliers_per_part)}')
            print(f'Time AGG1: {time.time() - checkpoint:.2f}s')
        checkpoint = time.time()

        # AGG2
        # avg_supplycost = partsupp_dao.avg_supplycost_by_part_size()
        # if verbose:
        #     print(f'AGG2) Avg supply cost by size groups: {len(avg_supplycost)}')
        #     print(f'Time AGG2: {time.time() - checkpoint:.2f}s')
