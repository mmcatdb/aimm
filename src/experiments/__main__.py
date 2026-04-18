import argparse
from core.utils import auto_close, data_size_quantity, pretty_print_int

NUM_RUNS = 1
# FIXME this
TEST_SCHEMA = 'todo'
SCALE = 1.0

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Test Postgres EDBT')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('check', help='Check database connections and sizes')

    args = parser.parse_args(rawArgs)

    if args.command == 'check':
        check_run()

def check_run():
    from core.config import Config
    from core.drivers import MongoDriver, Neo4jDriver, PostgresDriver

    config = Config.load()
    postgres = PostgresDriver(config.postgres, TEST_SCHEMA)
    mongo = MongoDriver(config.mongo, TEST_SCHEMA)
    neo4j = Neo4jDriver(config.neo4j, TEST_SCHEMA)

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
        print('Mongo')
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

if __name__ == '__main__':
    main()
