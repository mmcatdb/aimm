from common.config import Config
from common.driver_provider import DriverProvider
from datasets.tpch.query_engine import TpchQueryEngine

def main():
    options = ['postgres', 'mongo', 'neo4j']
    config = Config.load()
    dbs = DriverProvider.default(config)
    query_engine = TpchQueryEngine.create(dbs)

    best_time = float('inf')
    best_mapping = None

    for db_type1 in options:
        for db_type2 in options:
            for db_type3 in options:
                mapping = {
                    'customer': db_type1,
                    'orders': db_type2,
                    'lineitem': db_type3
                }

                execution_time = query_engine.measure_queries(mapping, verbose=False)

                if execution_time < best_time:
                    best_time = execution_time
                    best_mapping = mapping
                    print(f'New best time: {best_time} with mapping: {mapping}')

    print(f'Best time: {best_time} with mapping: {best_mapping}')

if __name__ == '__main__':
    main()
