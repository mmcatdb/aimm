import sys
import argparse
from common.config import Config, DatasetName
from common.drivers import PostgresDriver, Neo4jDriver, DriverType
from common.utils import auto_close, trim_to_block, exit_with_error
from common.explainers.common import OperatorNameFormatter
from common.database import Database
from datasets.databases import find_database, get_available_dataset_names
from latency_estimation.context import BaseContext

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Show a query plan visually.')
    subparsers = parser.add_subparsers(dest='database', required=True)

    _common_args(subparsers.add_parser(DriverType.POSTGRES.value), DriverType.POSTGRES)
    _common_args(subparsers.add_parser(DriverType.NEO4J.value), DriverType.NEO4J)

    args = parser.parse_args(rawArgs)
    type = DriverType(args.database)

    if type == DriverType.POSTGRES:
        _postgres_run(args)
    elif type == DriverType.MONGO:
        raise NotImplementedError('MongoDB is not supported yet.')
    elif type == DriverType.NEO4J:
        _neo4j_run(args)

def _common_args(parser: argparse.ArgumentParser, type: DriverType):
    parser.add_argument('dataset', nargs=1, choices=get_available_dataset_names(), help=f'Name of the dataset. Needed to select the database.')
    parser.add_argument('query', nargs='?', help='Query string or a test query ID (if such query exists). Read from stdin if omitted.')
    parser.add_argument('--tree', action=argparse.BooleanOptionalAction, default=True, help='Print the visual tree.')
    parser.add_argument('--json', action='store_true', help='Print the raw JSON.')
    parser.add_argument('--all-queries', action='store_true', help='Use all queries from the provided dataset. The query argument is ignored.')

    if type == DriverType.POSTGRES:
        parser.add_argument('--profile', action='store_true', help='Use EXPLAIN ANALYZE (actually runs the query; DML is rolled back).')
        parser.add_argument('--no-cache', action='store_true', help='Issue DISCARD ALL before running (requires --profile).')
    elif type == DriverType.NEO4J:
        parser.add_argument('--profile', action='store_true', help='Use PROFILE instead of EXPLAIN (actually runs the query; DML is rolled back).')

def _postgres_run(args: argparse.Namespace):
    from common.explainers.postgres_explainer import PostgresExplainer

    dataset = DatasetName(args.dataset[0])
    database = find_database(dataset, DriverType.POSTGRES)
    queries = _get_queries_from_input(args, database)

    config = Config.load()
    operators = _try_get_available_operators(config, database)
    driver = PostgresDriver(config.postgres, dataset.value)

    with auto_close(driver):
        explainer = PostgresExplainer(driver, operators)

        for query_id, query_content in queries:
            if args.all_queries:
                print(f'Query {query_id}:\n{trim_to_block(query_content)}\n')

            plan = explainer.fetch_plan(query_content, do_profile=args.profile, do_discard=args.no_cache)

            if args.tree:
                explainer.print_tree(plan)
            if args.json:
                print()
                explainer.print_json(plan)

    _try_print_missing_operators(operators)

def _neo4j_run(args: argparse.Namespace):
    from common.explainers.neo4j_explainer import Neo4jExplainer

    dataset = DatasetName(args.dataset[0])
    database = find_database(dataset, DriverType.NEO4J)
    queries = _get_queries_from_input(args, database)

    config = Config.load()
    operators = _try_get_available_operators(config, database)
    driver = Neo4jDriver(config.neo4j, dataset.value)

    with auto_close(driver):
        explainer = Neo4jExplainer(driver, operators)

        for query_id, query_content in queries:
            if args.all_queries:
                print(f'Query {query_id}:\n{trim_to_block(query_content)}\n')

            plan = explainer.fetch_plan(query_content, do_profile=args.profile)

            if args.tree:
                explainer.print_tree(plan)
            if args.json:
                print()
                explainer.print_json(plan)

    _try_print_missing_operators(operators)

def _get_queries_from_input(args: argparse.Namespace, database: Database) -> list[tuple[str, str]]:
    """Returns a list of (query_id, query_content) tuples. The query_id is empty if the query was provided directly rather than by ID."""
    if args.all_queries:
        return [(query.id, query.content) for query in database.get_test_queries()]

    query_or_id = _get_query_or_id_from_input(args)

    test_query = database.try_get_test_query(query_or_id)
    if test_query is not None:
        print(f'Found test query with ID "{test_query.id}":\n{trim_to_block(test_query.content)}\n')
        return [(test_query.id, test_query.content)]

    return [('', query_or_id)]

def _get_query_or_id_from_input(args: argparse.Namespace) -> str:
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

def _try_get_available_operators(config: Config, database: Database) -> OperatorNameFormatter | None:
    operator_list = BaseContext.load_available_operators(config, database)
    if operator_list is None:
        return None

    operators = OperatorNameFormatter(False)
    for operator in operator_list:
        operators.add(operator)

    return operators

def _try_print_missing_operators(operators: OperatorNameFormatter | None):
    if operators is None:
        return

    keys = operators.get_missing_keys()
    types = operators.get_missing_types()
    if keys:
        print('\nMissing operators:')
        for key in keys:
            print(f'  - {key}')

    if types:
        print('\nMissing operator types:')
        for type in types:
            print(f'  - {type}')

if __name__ == '__main__':
    main()
