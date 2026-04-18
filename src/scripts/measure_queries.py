import argparse
from core.config import Config
from core.driver_provider import DriverProvider
from core.query import DatabaseId, MeasuredQueries, QueryInstance, QueryRegistry, parse_database_id, save_measured
from core.utils import exit_with_exception
from core.dynamic_provider import get_dynamic_class_instance
from latency_estimation.class_provider import get_plan_extractor
from providers.path_provider import PathProvider

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Measure queries in a database.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()
    database_id = args.database_id[0]

    try:
        run(config, database_id, args.num_queries, args.num_runs)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('database_id',   nargs=1,                 help='Id of the database. Pattern: {driver_type}/{schema_name}-{scale}')
    parser.add_argument('--num-queries', type=int, required=True, help='Number of queries to collect.')
    parser.add_argument('--num-runs',    type=int, required=True, help='Number of executions per query for averaging.')

def run(config: Config, database_id: DatabaseId, num_queries: int, num_runs: int):
    dp = DriverProvider.default(config)
    pp = PathProvider(config)

    measured = measure_database(dp, database_id, num_queries, num_runs)

    save_measured(pp.measured(database_id, num_queries, num_runs), measured)

def measure_database(dp: DriverProvider, database_id: DatabaseId, num_queries: int, num_runs: int) -> MeasuredQueries:
    driver_type, schema, scale = parse_database_id(database_id)
    registry = get_dynamic_class_instance(QueryRegistry, driver_type, schema)
    queries = registry.generate_queries(scale, num_queries)

    return measure_database_queries(dp, database_id, queries, num_queries, num_runs)

def measure_database_queries(dp: DriverProvider, database_id: DatabaseId, queries: list[QueryInstance], num_queries: int, num_runs: int) -> MeasuredQueries:
    driver = dp.get(*parse_database_id(database_id))
    plan_extractor = get_plan_extractor(driver)
    items = plan_extractor.measure_and_explain_queries(queries, num_runs)
    global_stats = plan_extractor.collect_global_stats()

    return MeasuredQueries(items, global_stats, database_id, num_queries, num_runs)

if __name__ == '__main__':
    main()
