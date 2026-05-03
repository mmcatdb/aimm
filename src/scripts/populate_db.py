import argparse
from core.config import Config
from core.driver_provider import DriverProvider
from core.drivers import DriverType, PostgresDriver
from core.dynamic_provider import get_dynamic_class_instance
from core.loaders.base_loader import BaseLoader, save_populate_times
from core.query import create_schema_id, parse_database_id
from core.utils import auto_close, exit_with_exception
from providers.path_provider import PathProvider

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Load schema data into database.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()
    database_id = args.database_id[0]

    try:
        run(config, database_id, not args.no_reset)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('database_id', nargs=1, help='Id of the database. Pattern: {driver_type}/{schema_name}-{scale}')
    parser.add_argument('--no-reset', action='store_true', help='Skip clearing the database beforehand.')

def run(config: Config, database_id: str, do_reset: bool):
    pp = PathProvider(config)

    driver_type, schema, scale = parse_database_id(database_id)
    schema_id = create_schema_id(schema, scale)

    loader = get_dynamic_class_instance(BaseLoader, driver_type, schema)

    dp = DriverProvider.default(config)

    if driver_type == DriverType.POSTGRES:
        from core.loaders.postgres_loader import create_database_if_not_exists

        # Postgres databases have to be created explicitly, so let's do that.
        rootDriver = dp.get_by_name(PostgresDriver, config.postgres.root_database)
        database_name = dp.compute_database_name(schema, scale)
        with auto_close(rootDriver):
            create_database_if_not_exists(rootDriver, database_name)

    driver = dp.get(driver_type, schema, scale)

    with auto_close(driver):
        times = loader.run(driver, pp.imports(schema_id), do_reset)

    save_populate_times(pp.populate_times(database_id), times)

if __name__ == '__main__':
    main()
