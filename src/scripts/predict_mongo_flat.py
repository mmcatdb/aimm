import argparse
import sys

from core.config import Config
from core.driver_provider import DriverProvider
from core.drivers import DriverType, MongoDriver
from core.query import MongoQuery, parse_database_id
from core.utils import auto_close, exit_with_error, exit_with_exception, trim_to_block
from latency_estimation.dataset import parse_dataset_id
from latency_estimation.mongo.flat_model import load_flat_model
from latency_estimation.mongo.plan_extractor import PlanExtractor
from providers.path_provider import PathProvider


def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Predict MongoDB query latency from queryPlanner-only flat features.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    try:
        run(Config.load(), args)
    except Exception as e:
        exit_with_exception(e)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('model_id', nargs=1, help='Id of the flat model. Pattern: mongo/{model_name}.')
    parser.add_argument('database_id', nargs=1, help='Id of the database. Pattern: mongo/{schema_name}-{scale}.')
    parser.add_argument('query', nargs='?', help='Mongo query as JSON command. If omitted, read from stdin.')


def run(config: Config, args: argparse.Namespace):
    pp = PathProvider(config)
    model_id = args.model_id[0]
    database_id = args.database_id[0]

    model_driver_type, _ = parse_dataset_id(model_id)
    database_driver_type, schema, scale = parse_database_id(database_id)
    if model_driver_type != DriverType.MONGO or database_driver_type != DriverType.MONGO:
        raise ValueError('MongoDB flat-feature latency prediction requires mongo/{...} model and database ids.')

    query_text = _get_query(args)
    query = MongoQuery.parse(query_text)
    model = load_flat_model(pp.flat_model(model_id))

    dp = DriverProvider.default(config)
    driver = dp.get_typed(MongoDriver, schema, scale)
    plan_extractor = PlanExtractor(driver)

    with auto_close(driver):
        global_stats = plan_extractor.collect_global_stats()
        plan = plan_extractor.explain_query(query, do_profile=False)
        predicted = model.predict_plan(plan, global_stats)

    print(trim_to_block(query.serialize()))
    print(f'\nPredicted latency: {predicted:.2f} ms')
    print('Plan source: MongoDB explain verbosity "queryPlanner"; the query was not executed.')
    print('Global stats were collected once before prediction and reused as inference features.')


def _get_query(args: argparse.Namespace) -> str:
    if args.query:
        query = args.query
    elif not sys.stdin.isatty():
        query = sys.stdin.read().strip()
    else:
        print('Enter your Mongo query JSON (finish with Ctrl-D on a blank line):')
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
