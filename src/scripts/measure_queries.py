import argparse
import os
from core.config import Config
from core.driver_provider import DriverProvider
from core.query import DatabaseId, MeasuredQueriesPersistor, MeasurementConfig, MeasuredQueries, QueryInstance, QueryMeasurement, QueryRegistry, TQuery, load_measured, parse_database_id, print_warning, save_measured
from core.utils import ProgressTracker, exit_with_exception, plural
from core.dynamic_provider import get_dynamic_class_instance
from latency_estimation.class_provider import get_plan_extractor
from latency_estimation.plan_extractor import BasePlanExtractor
from providers.path_provider import PathProvider

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Measure queries in a database.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()
    database_id = args.database_id[0]
    mc = MeasurementConfig(
        num_queries=args.num_queries,
        num_runs=args.num_runs,
        allow_write=not args.no_write,
    )

    try:
        run(config, database_id, mc, not args.no_cache)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('database_id',   nargs=1,                 help='Id of the database. Pattern: {driver_type}/{schema_name}-{scale}')
    parser.add_argument('--num-queries', type=int, required=True, help='Number of queries to collect. At least one query will be generated for each query template.')
    parser.add_argument('--num-runs',    type=int, required=True, help='Number of executions per query for averaging.')
    parser.add_argument('--no-write',    action='store_true',     help='Disallow write queries (i.e., insert, update, delete).')
    parser.add_argument('--no-cache',    action='store_true',     help='Discard all previous measurements. Otherwise, only queries that have not been measured before will be executed and added to the output file.')

def run(config: Config, database_id: DatabaseId, mc: MeasurementConfig, use_cache: bool):
    dp = DriverProvider.default(config)
    pp = PathProvider(config)

    driver_type, schema, scale = parse_database_id(database_id)
    registry = get_dynamic_class_instance(QueryRegistry, driver_type, schema)
    queries = registry.generate_queries(scale, mc.num_queries, mc.allow_write)

    if len(queries) > mc.num_queries:
        print_warning(f'Number of generated queries ({len(queries)}) is greater than the requested --num-queries ({mc.num_queries}). Adjusting --num-queries to {len(queries)} to include all generated queries.')
        mc.num_queries = len(queries)

    driver = dp.get(*parse_database_id(database_id))
    plan_extractor = get_plan_extractor(driver)
    cache_path = pp.measured(database_id, mc)
    cached = get_or_create_initial_cache(cache_path, use_cache, plan_extractor, database_id, mc)

    missing_queries = [q for q in queries if q.id not in {m.id for m in cached.items}]
    if not missing_queries:
        print('All queries have already been measured in the cache. No new measurements needed.')
        return

    persistor = MeasuredQueriesPersistor.open(cache_path)
    try:
        measure_and_explain_queries(persistor, plan_extractor, missing_queries, mc.num_runs)
    finally:
        # This should flush all measurements even if keyboard interrupt or other exception happens during measurement.
        persistor.close()

def get_or_create_initial_cache(cache_path: str, use_cache: bool, plan_extractor: BasePlanExtractor[TQuery], database_id: DatabaseId, mc: MeasurementConfig) -> MeasuredQueries[TQuery]:
    if use_cache and os.path.exists(cache_path):
        try:
            cache = load_measured(cache_path)
            print(f'Loaded {plural(len(cache.items), "cached measurement")} from previous runs.')
            return cache
        except Exception as e:
            print_warning(f'Could not load cache from "{cache_path}". Starting with an empty cache.', e)

    global_stats = plan_extractor.collect_global_stats()
    cache = MeasuredQueries([], global_stats, database_id, mc)
    save_measured(cache_path, cache)

    return cache

def measure_and_explain_queries(persistor: MeasuredQueriesPersistor[TQuery], plan_extractor: BasePlanExtractor[TQuery], queries: list[QueryInstance[TQuery]], num_runs: int):
    progress = ProgressTracker.limited(len(queries))
    progress.start(f'Measuring {plural(len(queries), "query", "queries")} ({num_runs} runs each) ... ')

    invalid_queries = list[QueryInstance[TQuery]]()
    measured_count = 0

    for query in queries:
        is_exception = False
        try:
            measurement = plan_extractor.measure_and_explain_query(query, num_runs)
            persistor.append(measurement)
            measured_count += 1
        except KeyboardInterrupt:
            print('\nMeasurement interrupted by user. Stopping...')
            break
        except Exception as e:
            is_exception = True
            print()
            # Don't print the stack trace as we don't care about some random internals of the driver.
            print_warning(f'Could not execute query {query.label}.', e, suppress_stacktrace=True)
            print()
            invalid_queries.append(query)
        progress.track(force_print=is_exception)

    remaining_count = len(queries) - measured_count - len(invalid_queries)
    # If there was an interrupt, we don't want to print this.
    if remaining_count == 0:
        progress.finish()
        remaining_message = ''
    else:
        remaining_message = f' Remaining queries: {remaining_count}.'

    print(f'\nCollected {plural(measured_count, "measurement")} successfully.{remaining_message}')

    if invalid_queries:
        queries_str = '\n'.join(f'  {q.label}' for q in invalid_queries)
        print()
        print_warning(f'Failed to execute {plural(len(invalid_queries), "query", "queries")}:\n{queries_str}')

if __name__ == '__main__':
    main()
