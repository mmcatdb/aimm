import time
from typing_extensions import override
from search.query_engine import QueryEngine
from core.driver_provider import DriverProvider
from core.drivers import MongoDriver, Neo4jDriver, PostgresDriver
from .daos.tpch_dao import TpchDAO
from .daos.postgres_dao import TpchPostgresDAO
from .daos.mongo_dao import TpchMongoDAO
from .daos.neo4j_dao import TpchNeo4jDAO

SCHEMA = 'tpch' # FIXME Should be the same as the one used for the drivers/DAOs

class TpchQueryEngine(QueryEngine):
    def __init__(self, daos: dict[str, TpchDAO]):
        self.__daos = daos

    @staticmethod
    def create(dbs: DriverProvider):
        daos: dict[str, TpchDAO] = {
            'postgres': TpchPostgresDAO(dbs.get_by_name(PostgresDriver, SCHEMA)),
            'mongo': TpchMongoDAO(dbs.get_by_name(MongoDriver, SCHEMA)),
            'neo4j': TpchNeo4jDAO(dbs.get_by_name(Neo4jDriver, SCHEMA)),
        }

        return TpchQueryEngine(daos)

    @override
    def available_databases(self) -> list[str]:
        return list(self.__daos.keys())

    @override
    def measure_queries(self, mapping: dict[str, str], verbose=True) -> float:
        # Queries from: https://github.com/wsawa-q/evaluation-of-db-performance/blob/main/evaluation/database/mysql/queries.md

        self.__mapping = mapping

        start = checkpoint = time.time()

        if verbose:
            print(f'Customers stored in: {self.__mapping["customer"]}')
            print(f'Orders stored in: {self.__mapping["orders"]}')
            print(f'Lineitems stored in: {self.__mapping["lineitem"]}')
            print(f'Parts stored in: {self.__mapping["part"]}')
            print(f'Suppliers stored in: {self.__mapping["supplier"]}')
            print(f'PartSupp stored in: {self.__mapping["partsupp"]}')
            print('=' * 50)

        lineitem_dao = self.__get_dao_for_entity('lineitem')
        customer_dao = self.__get_dao_for_entity('customer')
        order_dao = self.__get_dao_for_entity('orders')

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
        customer_name='Customer#000000007'
        lineitems = self.__find_lineitems_for_customer(customer_name)
        if verbose:
            print(f'Found {len(lineitems)} lineitems for customer: {customer_name}')
            print(f'Time taken for Join: {time.time() - checkpoint:.2f} seconds')
            print('-' * 50)
        checkpoint = time.time()

        self.__run_part_supplier_queries(verbose=verbose)

        if verbose:
            print(f'Finished. Total time taken: {time.time() - start:.2f} seconds')

        return round(time.time() - start, 2)

    def __run_part_supplier_queries(self, verbose=True):
        part_dao = self.__get_dao_for_entity('part')
        supplier_dao = self.__get_dao_for_entity('supplier')
        partsupp_dao = self.__get_dao_for_entity('partsupp')
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

    def __find_lineitems_for_customer(self, customer_name):
        # Find the customer
        customer_dao = self.__get_dao_for_entity('customer')
        customers = customer_dao.find('customer', {'c_name': customer_name})
        if not customers:
            return []
        customer = customers[0]

        # Find all orders for the customer
        orders_dao = self.__get_dao_for_entity('orders')
        orders = orders_dao.find('orders', {'o_custkey': customer['c_custkey']})
        if not orders:
            return []

        # Get all order keys from the results
        order_keys = [order['o_orderkey'] for order in orders]

        # Find all lineitems for the given orders
        lineitem_dao = self.__get_dao_for_entity('lineitem')
        all_lineitems = lineitem_dao.find('lineitem', {'l_orderkey__in': order_keys})

        return all_lineitems

    def __get_dao_for_entity(self, entity: str):
        db_type = self.__mapping[entity]
        if not db_type:
            raise ValueError(f'Entity "{entity}" not found in schema mapping.')
        return self.__daos[db_type]
