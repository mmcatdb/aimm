import argparse
from common.utils import auto_close, data_size_quantity, pretty_print_int
from common.driver_provider import DatasetName
from latency_estimation.exceptions import NeuralUnitNotFoundException
from latency_estimation.common import NnOperator

NUM_RUNS = 1
# FIXME this
TEST_DATASET = DatasetName.EDBT

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Test Postgres EDBT')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('check', help='Check database connections and sizes')
    evaluate_args(subparsers.add_parser('evaluate', help='Evaluate models'))

    args = parser.parse_args(rawArgs)

    if args.command == 'check':
        check_run()
    elif args.command == 'evaluate':
        evaluate_run(args)

def check_run():
    from common.config import Config
    from common.drivers import MongoDriver, Neo4jDriver, PostgresDriver

    config = Config.load()
    postgres = PostgresDriver(config.postgres, TEST_DATASET.value)
    mongo = MongoDriver(config.mongo)
    neo4j = Neo4jDriver(config.neo4j, TEST_DATASET.value)

    print('Checking database connections and sizes ...')

    with auto_close(postgres):
        print('Postgres')
        print('- connection ... ', end='')
        connection = postgres.get_connection()
        postgres.put_connection(connection)
        print('OK')

        print('- records    ... ', end='')
        count = postgres.query_record_count()
        print(pretty_print_int(count))

        print('- size       ... ', end='')
        size = postgres.query_total_size()
        print(data_size_quantity.pretty_print(size))

    with auto_close(mongo):
        print('MongoDB')
        print('- connection ... ', end='')
        mongo.database().list_collection_names()
        print('OK')

        print('- records    ... ', end='')
        count = mongo.query_record_count()
        print(pretty_print_int(count))

        print('- size       ... ', end='')
        size = mongo.query_total_size()
        print(data_size_quantity.pretty_print(size))

    with auto_close(neo4j):
        print('Neo4j')
        print('- connection ... ', end='')
        neo4j.verify()
        print('OK')

        print('- records    ... ', end='')
        count = neo4j.query_record_count()
        print(pretty_print_int(count))

    print('Done.')

def evaluate_args(parser: argparse.ArgumentParser):
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to trained model')
    parser.add_argument('--database', '-d', type=str, required=True, help='Either "postgres" or "neo4j"')

def evaluate_run(args: argparse.Namespace):
    print(f'Evaluating {args.database} model\n')

    if args.database == 'postgres':
        evaluate_postgres(args.checkpoint)
    elif args.database == 'neo4j':
        evaluate_neo4j(args.checkpoint)
    else:
        print(f'Unsupported database type: {args.database}')

def evaluate_postgres(checkpoint: str):
    from datasets.edbt.postgres_database import EdbtPostgresDatabase
    from latency_estimation.postgres.context import PostgresContext
    from latency_estimation.postgres.latency_estimator import LatencyEstimator

    missing_operators: set[str] = set()

    ctx = PostgresContext.create(database=EdbtPostgresDatabase())
    with auto_close(ctx):
        model = ctx.load_model(checkpoint)
        estimator = LatencyEstimator(ctx.extractor, model)

        for query in ctx.database.get_test_queries():
            try:
                print(f'Executing query {query.label()}...')
                estimated, _ = estimator.estimate(query.content)
                actual, _, _, num_results = ctx.extractor.measure_query(query.content, num_runs=NUM_RUNS)
                print_query_results(num_results, estimated, actual)
            except NeuralUnitNotFoundException as e:
                missing_operators.add(e.operator.key())
                print(f'Error: {e}\n')
            except Exception as e:
                print(f'Error: {e}\n')

    try_print_missing_operators(missing_operators, model.get_units())

def evaluate_neo4j(checkpoint: str):
    from datasets.edbt.neo4j_database import EdbtNeo4jDatabase
    from latency_estimation.neo4j.context import Neo4jContext
    from latency_estimation.neo4j.latency_estimator import LatencyEstimator

    missing_operators: set[str] = set()

    ctx = Neo4jContext.create(database=EdbtNeo4jDatabase())
    with auto_close(ctx):
        model = ctx.load_model(checkpoint)
        estimator = LatencyEstimator(ctx.extractor, model)

        for query in ctx.database.get_test_queries():
            try:
                print(f'Executing query {query.label()}...')
                estimated, _ = estimator.estimate(query.content)
                actual, _, num_results = ctx.extractor.measure_query(query.content, num_runs=NUM_RUNS)
                print_query_results(num_results, estimated, actual)
            except NeuralUnitNotFoundException as e:
                missing_operators.add(e.operator.key())
                print(f'Error: {e}\n')
            except Exception as e:
                print(f'Error: {e}\n')

        try_print_missing_operators(missing_operators, model.get_units())

def print_query_results(num_results: int, estimated: float, actual: float):
    print(f'Returned {num_results} rows. Estimated: {estimated * 1000:.2f} ms, Actual: {actual * 1000:.2f} ms\n')

def try_print_missing_operators(missing: set[str], available: list[NnOperator]):
    if not missing:
        return

    print('Missing neural units for the following operators:')
    for operator in missing:
        print(f'  - {operator}')
    print('Available operators:')
    for operator in available:
        print(f'  - {operator.type}')

if __name__ == '__main__':
    main()
