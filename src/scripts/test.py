import os
import argparse
import json
from typing_extensions import deprecated
from core.config import Config
from core.dynamic_provider import get_dynamic_class_instance
from core.utils import JsonEncoder, exit_with_error, exit_with_exception, print_warning
from core.query import QueryInstance, QueryRegistry, parse_database_id, parse_query
from latency_estimation.class_provider import get_model_evaluator, get_plan_extractor
from latency_estimation.feature_extractor import load_feature_extractor
from latency_estimation.latency_estimator import LatencyEstimator
from providers.contex import Context

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Test models on queries.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    ctx = Context.default()
    database_id = args.database_id[0]

    try:
        run(ctx, args, database_id)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('database_id',        nargs=1,                                   help='Id of the database. Pattern: {driver_type}/{schema_name}-{scale}')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,                   help='Path to model checkpoint.')
    parser.add_argument('--feature-extractor', '-f', type=str, required=True,            help='Path to the feature extractor.')
    parser.add_argument('--num-runs',         type=int, default=3,                       help='Number of executions per query for averaging.')
    parser.add_argument('--num-queries',      type=int, default=0,                       help='Number of built-in query instances to generate. Defaults to one per template.')
    parser.add_argument('--query', '-q',      type=str, action='append', dest='queries', help='Additional query to test (can be used multiple times). Disables built-in test queries.')

    # TODO This is only for postgres
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots.')

def run(ctx: Context, args: argparse.Namespace, database_id: str):
    driver_type, schema, scale = parse_database_id(database_id)

    queries = get_query_instances(args)

    estimator = LatencyEstimator()

    model = ctx.mp.load_model(args.checkpoint)
    feature_extractor = load_feature_extractor(args.feature_extractor)
    estimator.register_driver_type(driver_type, feature_extractor, model)

    driver = ctx.dp.get(driver_type, schema, scale)
    plan_extractor = get_plan_extractor(driver)
    estimator.register_database(database_id, plan_extractor)

    evaluator = get_model_evaluator(driver_type, estimator)
    # FIXME Unify
    results = evaluator.evaluate_queries(queries, args.num_runs)
    evaluator.print_summary(results)

    save_results(ctx.config, results)

    if not args.no_plots:
        try:
            plot_path = os.path.join(ctx.config.results_directory, 'evaluation_plots.png')
            evaluator.plot_results(results, save_path=plot_path)
        except Exception as e:
            print_warning('Could not generate plots.', e)

@deprecated('unify')
def get_query_instances(args: argparse.Namespace) -> list[QueryInstance]:
    database_id = args.database_id[0]
    query_strings = args.queries or []

    driver_type, schema, scale = parse_database_id(database_id)

    if query_strings:
        queries = list[QueryInstance]()
        for i, content_str in enumerate(query_strings):
            try:
                content = parse_query(driver_type, content_str)
                queries.append(QueryInstance.create_custom(driver_type, schema, scale, i, content))
            except Exception as e:
                print_warning(f'Could not parse query {i + 1}. Skipping.', e)

        print(f'\nAdded {len(queries)} custom query/queries')
    else:
        print('\nGenerating test queries...')
        registry = get_dynamic_class_instance(QueryRegistry, driver_type, schema)
        queries = registry.generate_queries(scale, args.num_queries)

    if not queries:
        exit_with_error('No queries to test. Provide queries with --query or use the built-in test queries.')

    print(f'Total queries to test: {len(queries)}')

    return queries

def save_results(config: Config, results: list):
    path = os.path.join(config.results_directory, 'evaluation_results.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f'\nSaving results to {path}...')
    with open(path, 'w') as file:
        json.dump(results, file, indent=4, cls=JsonEncoder)

if __name__ == '__main__':
    main()
