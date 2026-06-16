import argparse
import os
import sys

from bson import json_util

from core.config import Config
from core.driver_provider import DriverProvider
from core.drivers import DriverType, MongoDriver
from core.query import MongoDeleteQuery, MongoInsertQuery, MongoQuery, MongoUpdateQuery, parse_database_id
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
    parser.add_argument('--collect-global-stats', action='store_true', help='Collect and cache Mongo field stats if no precomputed cache exists. Intended for data prep, not per-query inference.')


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
        global_stats = _load_cached_global_stats(pp, database_id)
        if global_stats is None:
            if not args.collect_global_stats:
                exit_with_error(
                    f'No cached Mongo global stats with field distributions found for "{database_id}". '
                    'Recreate the Mongo flat dataset/measurements first, or run this command once with '
                    '--collect-global-stats during data prep.'
                )
            global_stats = plan_extractor.collect_global_stats()
            _save_cached_global_stats(pp, database_id, global_stats)
        plan = plan_extractor.explain_query(query, _is_write_query(query), do_profile=False)
        predicted = model.predict_plan(plan, global_stats)

    print(trim_to_block(query.serialize()))
    print(f'\nPredicted latency: {predicted:.2f} ms')
    print('Plan source: MongoDB explain verbosity "queryPlanner"; the query was not executed.')
    print('Global stats were loaded from the precomputed cache and reused as inference features.')


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

def _is_write_query(query: MongoQuery) -> bool:
    return isinstance(query, (MongoUpdateQuery, MongoDeleteQuery, MongoInsertQuery))


def _load_cached_global_stats(pp: PathProvider, database_id: str) -> dict | None:
    path = pp.global_stats(database_id)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        stats = json_util.loads(file.read())
    if not _has_mongo_field_stats(stats):
        return None
    return stats


def _save_cached_global_stats(pp: PathProvider, database_id: str, global_stats: dict) -> None:
    path = pp.global_stats(database_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(json_util.dumps(global_stats))


def _has_mongo_field_stats(global_stats: dict) -> bool:
    if not isinstance(global_stats, dict) or not global_stats:
        return False
    collection_stats = [stats for stats in global_stats.values() if isinstance(stats, dict)]
    return bool(collection_stats) and all(PlanExtractor.FIELD_STATS_KEY in stats for stats in collection_stats)


if __name__ == '__main__':
    main()
