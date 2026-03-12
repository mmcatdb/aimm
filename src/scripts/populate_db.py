import argparse
from common.config import Config
from common.driver_provider import DriverProvider, DatasetName, dataset_import_directory
from common.drivers import DriverType, PostgresDriver, MongoDriver, Neo4jDriver
from common.utils import auto_close
from datasets.databases import get_available_dataset_names

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Neo4j QPP-Net')
    subparsers = parser.add_subparsers(dest='command', required=True)

    common_args(subparsers.add_parser(DriverType.POSTGRES.value, help='Load data into a Postgres database.'))
    common_args(subparsers.add_parser(DriverType.NEO4J.value, help='Load data into a Neo4j database.'))

    args = parser.parse_args(rawArgs)
    driver = DriverType(args.command)

    ctx = Context()
    ctx.setup(args)

    if driver == DriverType.POSTGRES:
        populate_postgres(ctx)
    elif driver == DriverType.MONGO:
        populate_mongo(ctx)
    elif driver == DriverType.NEO4J:
        populate_neo4j(ctx)

def common_args(parser: argparse.ArgumentParser):
    parser.add_argument('dataset', nargs=1, choices=get_available_dataset_names(), help=f'Name of the dataset.')
    parser.add_argument('--import-dir', type=str, default=None, help='Path to the directory containing the input files. If not specified, defaults to "{IMPORT_DIRECTORY}/<dataset>".')
    parser.add_argument('--no-reset', type=bool, default=False, help='Skip clearing the database beforehand.')

class Context:
    def setup(self, args: argparse.Namespace):
        self.dataset = DatasetName(args.dataset[0])
        self.database_name = self.dataset.value
        self.config = Config.load()
        self.dbs = DriverProvider.default(self.config)
        self.do_reset = not args.no_reset
        self.import_directory = args.import_dir or dataset_import_directory(self.config, self.dataset)

def populate_postgres(ctx: Context):
    from common.daos.postgres_dao import PostgresDAO

    # Postgres databases have to be created manually.
    rootDriver = ctx.dbs.get_for_database(ctx.config.postgres.root_database, PostgresDriver)
    with auto_close(rootDriver):
        rootDao = PostgresDAO(rootDriver)
        rootDao.create_database_if_not_exists(ctx.database_name)

    driver = ctx.dbs.get_for_database(ctx.database_name, PostgresDriver)
    with auto_close(driver):
        if ctx.dataset == DatasetName.EDBT:
            from datasets.edbt.postgres_loader import EdbtPostgresLoader
            loader = EdbtPostgresLoader(ctx.config, driver)
        elif ctx.dataset == DatasetName.TPCH:
            from datasets.tpch.postgres_loader import TpchPostgresLoader
            loader = TpchPostgresLoader(ctx.config, driver)
        else:
            raise ValueError(f'Unsupported dataset: {ctx.dataset}')

        loader.run(ctx.import_directory, ctx.do_reset)

def populate_mongo(ctx: Context):
    raise NotImplementedError('MongoDB loading is not implemented yet.')

def populate_neo4j(ctx: Context):
    driver = ctx.dbs.get_for_database(ctx.database_name, Neo4jDriver)
    with auto_close(driver):
        if ctx.dataset == DatasetName.EDBT:
            from datasets.edbt.neo4j_loader import EdbtNeo4jLoader
            loader = EdbtNeo4jLoader(ctx.config, driver)
        elif ctx.dataset == DatasetName.TPCH:
            from datasets.tpch.neo4j_loader import TpchNeo4jLoader
            loader = TpchNeo4jLoader(ctx.config, driver)
        else:
            raise ValueError(f'Unsupported dataset: {ctx.dataset}')

        loader.run(ctx.import_directory, ctx.do_reset)

if __name__ == '__main__':
    main()
