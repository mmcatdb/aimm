import argparse
import sys

from core.config import Config
from core.driver_provider import DriverProvider
from core.drivers import DriverType, PostgresDriver
from core.explainers.postgres_explainer import PostgresExplainer
from core.query import DatabaseId, parse_database_id
from core.utils import auto_close, exit_with_error, exit_with_exception, trim_to_block
from latency_estimation.dataset import parse_dataset_id
from latency_estimation.postgres.flat_model import load_flat_model
from providers.path_provider import PathProvider


def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Predict PostgreSQL query latency from EXPLAIN-only flat features.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    try:
        run(Config.load(), args)
    except Exception as e:
        exit_with_exception(e)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('model_id', nargs=1, help='Id of the flat model. Pattern: postgres/{model_name}.')
    parser.add_argument('database_id', nargs=1, help='Id of the database. Pattern: postgres/{schema_name}-{scale}.')
    parser.add_argument('query', nargs='?', help='SQL query. If omitted, read from stdin.')


def run(config: Config, args: argparse.Namespace):
    pp = PathProvider(config)
    model_id = args.model_id[0]
    database_id = args.database_id[0]

    model_driver_type, _ = parse_dataset_id(model_id)
    database_driver_type, schema, scale = parse_database_id(database_id)
    if model_driver_type != DriverType.POSTGRES or database_driver_type != DriverType.POSTGRES:
        raise ValueError('Flat-feature latency prediction is currently implemented only for PostgreSQL.')

    query = _get_query(args)
    model = load_flat_model(pp.flat_model(model_id))

    dp = DriverProvider.default(config)
    driver = dp.get_typed(PostgresDriver, schema, scale)
    explainer = PostgresExplainer(driver, operators=None)

    with auto_close(driver):
        plan = explainer.fetch_plan(query, do_profile=False)
        predicted = model.predict_plan(plan)

    print(trim_to_block(query))
    print(f'\nPredicted latency: {predicted:.2f} ms')
    print('Plan source: EXPLAIN without ANALYZE; the query was not executed.')


def _get_query(args: argparse.Namespace) -> str:
    if args.query:
        query = args.query
    elif not sys.stdin.isatty():
        query = sys.stdin.read().strip()
    else:
        print('Enter your query (finish with Ctrl-D on a blank line):')
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        query = '\n'.join(lines).strip()

    if not query:
        exit_with_error('No query provided.')
    return query


if __name__ == '__main__':
    main()
