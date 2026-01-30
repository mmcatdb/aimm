from common.config import Config
from common.database_provider import DatabaseProvider
from query_engine import QueryEngine

def main():
    config = Config.load()
    dbs = DatabaseProvider.default(config)
    # TODO
    schema_mapping = {
        'region': 'postgres',
        'nation': 'postgres',
        'customer': 'postgres',
        'orders': 'postgres',
        'lineitem': 'postgres',
        'part': 'postgres',
        'supplier': 'postgres',
        'partsupp': 'postgres',
    }
    engine = QueryEngine(dbs, schema_mapping)

    try:
        engine.run_queries(f'Customer#000000007')
        # print('--- Finding a customer ---')
        # customer = engine.find('customer', {'c_custkey': 1})
        # print(customer)
        # print('')


        # print('--- Finding number of orders with status F ---')
        # orders = engine.find('orders', {'o_orderstatus': 'F'})
        # print(f'Found {len(orders)} orders with status 'F'')
        # print('')


        # print('--- Lineitems for a customer ---')
        # lineitems = engine.find_lineitems_for_customer('Customer#000000001')
        # print('--- Results ---')
        # for item in lineitems:
        #     print(item)

    finally:
        print('\nDisconnected from all databases.')

if __name__ == '__main__':
    main()
