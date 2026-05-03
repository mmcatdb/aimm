import sys
import argparse
from typing_extensions import deprecated
from core.config import Config
from core.driver_provider import DriverProvider
from core.drivers import PostgresDriver, Neo4jDriver, DriverType
from core.dynamic_provider import get_dynamic_class_instance
from core.explainers.neo4j_explainer import Neo4jExplainer
from core.explainers.postgres_explainer import PostgresExplainer
from core.utils import auto_close, exit_with_exception, trim_to_block, exit_with_error
from core.explainers.common import OperatorNameFormatter
from core.query import QueryRegistry, QueryInstance, parse_database_id, DatabaseId
from latency_estimation.dataset import try_load_available_operators
from providers.path_provider import PathProvider

SCALE = 1.0  # FIXME

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Show a query plan visually.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    database_id = args.database_id[0]

    try:
        run(Config.load(), args, database_id)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('database_id', nargs=1, help='Id of the database. Pattern: {driver_type}/{schema_name}-{scale}')
    parser.add_argument('query', nargs='?', help='Query string or a query template name (if such query exists). Read from stdin if omitted.')
    parser.add_argument('--tree', action=argparse.BooleanOptionalAction, default=True, help='Print the visual tree.')
    parser.add_argument('--json', action='store_true', help='Print the raw JSON.')
    parser.add_argument('--all-queries', action='store_true', help='Use all queries from the provided schema. The query argument is ignored.')
    parser.add_argument('--operators', type=str, help='Id of a dataset to load the available operators from. Pattern: {driver_type}/{dataset_name}. If not provided, no operator information will be displayed.')
    parser.add_argument('--profile', action='store_true', help='Use EXPLAIN ANALYZE / PROFILE.')

def run(config: Config, args: argparse.Namespace, database_id: DatabaseId):
    pp = PathProvider(config)

    driver_type, schema, scale = parse_database_id(database_id)
    registry = get_dynamic_class_instance(QueryRegistry, driver_type, schema)
    queries = _get_queries_from_input(args, registry, scale)

    operators = _try_get_available_operators(pp.operators(args.operators)) if args.operators else None
    driver, explainer = _get_explainer(config, database_id, operators)

    with auto_close(driver):
        for query in queries:
            if args.all_queries:
                print(f'{query.label}:')

            print(trim_to_block(str(query.content)) + '\n')

            plan = explainer.fetch_plan(query.content, do_profile=args.profile)

            if args.tree:
                explainer.print_tree(plan)
            if args.json:
                print()
                explainer.print_json(plan)

    _try_print_missing_operators(operators)

@deprecated('unify')
def _get_queries_from_input(args: argparse.Namespace, registry: QueryRegistry, scale: float) -> list[QueryInstance]:
    """Returns a list of (query_id, query_content) tuples. The query_id is empty if the query was provided directly rather than by ID."""
    if args.all_queries:
        return registry.generate_queries(scale, 0)

    content_or_name = _get_query_content_or_name_from_input(args)

    template = registry.get_template(content_or_name)
    if template is not None:
        print(f'Found query template with name "{template.name}".')
        return [template.generate(scale, 0)]

    return [QueryInstance.create_custom(registry.driver, registry.schema, scale, 0, content_or_name)]

@deprecated('unify')
def _get_query_content_or_name_from_input(args: argparse.Namespace) -> str:
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

def _try_get_available_operators(path: str) -> OperatorNameFormatter | None:
    operator_list = try_load_available_operators(path)
    if operator_list is None:
        return None

    operators = OperatorNameFormatter(False)
    for operator in operator_list:
        operators.add(operator)

    return operators

def _get_explainer(config: Config, database_id: DatabaseId, operators: OperatorNameFormatter | None) -> tuple[PostgresDriver, PostgresExplainer] | tuple[Neo4jDriver, Neo4jExplainer]:
    driver_type, schema, scale = parse_database_id(database_id)
    dp = DriverProvider.default(config)

    if driver_type == DriverType.POSTGRES:
        from core.explainers.postgres_explainer import PostgresExplainer
        driver = dp.get_typed(PostgresDriver, schema, scale)
        return driver, PostgresExplainer(driver, operators)
    elif driver_type == DriverType.MONGO:
        raise NotImplementedError('MongoDB is not supported yet.')
    elif driver_type == DriverType.NEO4J:
        from core.explainers.neo4j_explainer import Neo4jExplainer
        driver = dp.get_typed(Neo4jDriver, schema, scale)
        return driver, Neo4jExplainer(driver, operators)
    else:
        raise ValueError(f'Unsupported driver type: {driver_type.value}')

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
