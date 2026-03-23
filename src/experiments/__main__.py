import argparse
from common.utils import auto_close, data_size_quantity, pretty_print_int, print_warning, print_info, exit_with_error
from common.config import DatasetName
from common.drivers import DriverType
from common.nn_operator import NnOperator
from latency_estimation.exceptions import NeuralUnitNotFoundException

NUM_RUNS = 1
# FIXME this
TEST_DATASET = DatasetName.EDBT

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Test Postgres EDBT')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('check', help='Check database connections and sizes')
    test_args(subparsers.add_parser('test', help='Test models'))

    args = parser.parse_args(rawArgs)

    if args.command == 'check':
        check_run()
    elif args.command == 'test':
        test_run(args)

def check_run():
    from common.config import Config
    from common.drivers import MongoDriver, Neo4jDriver, PostgresDriver

    config = Config.load()
    postgres = PostgresDriver(config.postgres, TEST_DATASET.value)
    mongo = MongoDriver(config.mongo, TEST_DATASET.value)
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

AVAILABLE_DATABASES = [type.value for type in [DriverType.POSTGRES, DriverType.NEO4J]]

def test_args(parser: argparse.ArgumentParser):
    parser.add_argument('database', nargs=1, choices=AVAILABLE_DATABASES, help='Type of database to test.')
    # The checkpoint is required since we usually want to test on a different dataset than we trained on, so we can't rely on the default checkpoint path.
    parser.add_argument('--checkpoint', '-c', type=str, required=True,     help='Path to model checkpoint.')

def test_run(args: argparse.Namespace):
    type = DriverType(args.database[0])

    print(f'Testing {type.value} model')

    if type == DriverType.POSTGRES:
        test_postgres(args.checkpoint)
    elif type == DriverType.NEO4J:
        test_neo4j(args.checkpoint)
    else:
        print(f'Unsupported database type: {type.value}')

def test_postgres(checkpoint: str):
    from latency_estimation.postgres.context import PostgresContext
    from latency_estimation.postgres.latency_estimator import LatencyEstimator

    missing_operators: set[str] = set()

    ctx = PostgresContext.create(dataset=TEST_DATASET)
    with auto_close(ctx):
        model = ctx.load_model(checkpoint)
        estimator = LatencyEstimator(ctx.extractor, model)

        for query in ctx.database().get_query_defs():
            try:
                print(f'Executing query {query.label()}...')
                content = query.generate()
                estimated, _ = estimator.estimate(content)
                actual, _, num_results = ctx.extractor.measure_query(content, num_runs=NUM_RUNS)
                print_query_results(num_results, estimated, actual)
            except NeuralUnitNotFoundException as e:
                missing_operators.add(e.operator.key())
                print_warning(str(e))
            except Exception as e:
                print_warning('Could not execute query.', e)

            print()

    try_print_missing_operators(missing_operators, model.get_units())

def test_neo4j(checkpoint: str):
    from latency_estimation.neo4j.context import Neo4jContext
    from latency_estimation.neo4j.latency_estimator import LatencyEstimator

    missing_operators: set[str] = set()

    ctx = Neo4jContext.create(dataset=TEST_DATASET)
    with auto_close(ctx):
        model = ctx.load_model(checkpoint)
        estimator = LatencyEstimator(ctx.extractor, model)

        for query in ctx.database().get_query_defs():
            try:
                print(f'Executing query {query.label()}...')
                content = query.generate()
                estimated, _ = estimator.estimate(content)
                actual, _, num_results = ctx.extractor.measure_query(content, num_runs=NUM_RUNS)
                print_query_results(num_results, estimated, actual)
            except NeuralUnitNotFoundException as e:
                missing_operators.add(e.operator.key())
                print_warning(str(e))
            except Exception as e:
                print_warning('Could not execute query.', e)

            print()

        try_print_missing_operators(missing_operators, model.get_units())

def print_query_results(num_results: int, estimated: float, actual: float):
    print(f'Returned {num_results} rows. Estimated: {estimated:.2f} ms, Actual: {actual:.2f} ms.')

def try_print_missing_operators(missing: set[str], available: list[NnOperator]):
    if not missing:
        return

    print('Missing neural units for the following operators:')
    for operator in missing:
        print(f'  - {operator}')
    print('\nAvailable operators:')
    for operator in available:
        print(f'  - {operator.key()}')

if __name__ == '__main__':
    main()
