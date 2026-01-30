from common.config import Config
from common.database_provider import DatabaseProvider
from query_engine import QueryEngine

def test_mapping_performance(dbs: DatabaseProvider, mapping: dict[str, str]) -> float:
    query_engine = QueryEngine(dbs, schema_mapping=mapping)
    return query_engine.run_queries(verbose=False)

def main():
    options = ['postgres', 'mongo', 'neo4j']
    config = Config.load()
    dbs = DatabaseProvider.default(config)

    best_time = float('inf')
    best_mapping = None

    for db_type1 in options:
        for db_type2 in options:
            for db_type3 in options:
                schema_mapping = {
                    'customer': db_type1,
                    'orders': db_type2,
                    'lineitem': db_type3
                }
                execution_time = test_mapping_performance(dbs, schema_mapping)
                if execution_time < best_time:
                    best_time = execution_time
                    best_mapping = schema_mapping
                    print(f'New best time: {best_time} with mapping: {schema_mapping}')

    print(f'Best time: {best_time} with mapping: {best_mapping}')

if __name__ == '__main__':
    main()
