import sys
import argparse
from common.config import Config, DatasetName
from common.drivers import PostgresDriver, Neo4jDriver, DriverType
from common.utils import auto_close, trim_to_block, exit_with_error
from datasets.databases import find_database, get_available_dataset_names

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Show a query plan visually.')
    subparsers = parser.add_subparsers(dest='database', required=True)

    common_args(subparsers.add_parser(DriverType.POSTGRES.value), DriverType.POSTGRES)
    common_args(subparsers.add_parser(DriverType.NEO4J.value), DriverType.NEO4J)

    args = parser.parse_args(rawArgs)
    type = DriverType(args.database)

    if type == DriverType.POSTGRES:
        postgres_run(args)
    elif type == DriverType.MONGO:
        raise NotImplementedError('MongoDB is not supported yet.')
    elif type == DriverType.NEO4J:
        neo4j_run(args)

def common_args(parser: argparse.ArgumentParser, type: DriverType):
    parser.add_argument('dataset', nargs=1, choices=get_available_dataset_names(), help=f'Name of the dataset. Needed to select the database.')
    parser.add_argument('query', nargs='?', help='Query string or a test query ID (if such query exists). Read from stdin if omitted.')
    parser.add_argument('--tree', action=argparse.BooleanOptionalAction, default=True, help='Print the visual tree.')
    parser.add_argument('--json', action='store_true', help='Print the raw JSON.')

    if type == DriverType.POSTGRES:
        parser.add_argument('--profile', action='store_true', help='Use EXPLAIN ANALYZE (actually runs the query; DML is rolled back).')
        parser.add_argument('--no-cache', action='store_true', help='Issue DISCARD ALL before running (requires --profile).')
    elif type == DriverType.NEO4J:
        parser.add_argument('--profile', action='store_true', help='Use PROFILE instead of EXPLAIN (actually runs the query; DML is rolled back).')

def postgres_run(args: argparse.Namespace):
    from common.explainers.postgres_explainer import PostgresExplainer

    dataset = DatasetName(args.dataset[0])
    query = get_query_from_input(args, DriverType.POSTGRES, dataset)

    config = Config.load()
    driver = PostgresDriver(config.postgres, dataset.value)

    with auto_close(driver):
        explainer = PostgresExplainer(driver)

        plan = explainer.fetch_plan(query, do_profile=args.profile, do_discard=args.no_cache)

        if args.tree:
            explainer.print_tree(plan)
        if args.json:
            print()
            explainer.print_json(plan)

def neo4j_run(args: argparse.Namespace):
    from common.explainers.neo4j_explainer import Neo4jExplainer

    dataset = DatasetName(args.dataset[0])
    query = get_query_from_input(args, DriverType.NEO4J, dataset)

    config = Config.load()
    driver = Neo4jDriver(config.neo4j, dataset.value)

    with auto_close(driver):
        explainer = Neo4jExplainer(driver)

        plan = explainer.fetch_plan(query, do_profile=args.profile)

        if args.tree:
            explainer.print_tree(plan)
        if args.json:
            print()
            explainer.print_json(plan)

def get_query_from_input(args: argparse.Namespace, driver: DriverType, dataset: DatasetName) -> str:
    query_or_id = get_query_or_id_from_input(args)
    database = find_database(dataset, driver)

    test_query = database.try_get_test_query(query_or_id)
    if test_query is not None:
        print(f'Found test query with ID "{test_query.id}":\n{trim_to_block(test_query.content)}\n')
        return test_query.content

    return query_or_id

def get_query_or_id_from_input(args: argparse.Namespace) -> str:
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
