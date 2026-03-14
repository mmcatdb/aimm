import argparse
from common.config import Config
from common.driver_provider import DriverProvider, DatasetName, dataset_import_directory
from common.drivers import DriverType, PostgresDriver, MongoDriver, Neo4jDriver
from common.utils import auto_close
from datasets.databases import get_available_dataset_names

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Load data into database.')
    subparsers = parser.add_subparsers(dest='database', required=True)

    common_args(subparsers.add_parser(DriverType.POSTGRES.value))
    common_args(subparsers.add_parser(DriverType.MONGO.value))
    common_args(subparsers.add_parser(DriverType.NEO4J.value))

    args = parser.parse_args(rawArgs)
    type = DriverType(args.database)

    ctx = Context()
    ctx.setup(args)

    if type == DriverType.POSTGRES:
        populate_postgres(ctx)
    elif type == DriverType.MONGO:
        populate_mongo(ctx)
    elif type == DriverType.NEO4J:
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

    # Postgres databases have to be created explicitly, so let's do that.
    rootDriver = ctx.dbs.get_for_database(ctx.config.postgres.root_database, PostgresDriver)
    with auto_close(rootDriver):
        rootDao = PostgresDAO(rootDriver)
        rootDao.create_database_if_not_exists(ctx.database_name)

    driver = ctx.dbs.get_for_database(ctx.database_name, PostgresDriver)
    with auto_close(driver):
        if ctx.dataset == DatasetName.EDBT:
            from datasets.edbt.postgres_loader import EdbtPostgresLoader
            loader = EdbtPostgresLoader(driver)
        elif ctx.dataset == DatasetName.TPCH:
            from datasets.tpch.postgres_loader import TpchPostgresLoader
            loader = TpchPostgresLoader(driver)
        else:
            raise ValueError(f'Unsupported dataset: {ctx.dataset}')

        loader.run(ctx.import_directory, ctx.do_reset)

def populate_mongo(ctx: Context):
    # Mongo databases are created automatically by writing in them. So, nothing to do here.

    driver = ctx.dbs.get_for_database(ctx.database_name, MongoDriver)
    with auto_close(driver):
        if ctx.dataset == DatasetName.EDBT:
            raise NotImplementedError('MongoDB loading is not implemented yet for EDBT.')
            # from datasets.edbt.mongo_loader import EdbtMongoLoader
            # loader = EdbtMongoLoader(driver)
        elif ctx.dataset == DatasetName.TPCH:
            from datasets.tpch.mongo_loader import TpchMongoLoader
            loader = TpchMongoLoader(driver)
        else:
            raise ValueError(f'Unsupported dataset: {ctx.dataset}')

        loader.run(ctx.import_directory, ctx.do_reset)

def populate_neo4j(ctx: Context):
    # Neo4j databases have to run in separate containers (the free edition doesn't support multiple databases). So, nothing to do here.

    driver = ctx.dbs.get_for_database(ctx.database_name, Neo4jDriver)
    with auto_close(driver):
        if ctx.dataset == DatasetName.EDBT:
            from datasets.edbt.neo4j_loader import EdbtNeo4jLoader
            loader = EdbtNeo4jLoader(driver)
        elif ctx.dataset == DatasetName.TPCH:
            from datasets.tpch.neo4j_loader import TpchNeo4jLoader
            loader = TpchNeo4jLoader(driver)
        else:
            raise ValueError(f'Unsupported dataset: {ctx.dataset}')

        loader.run(ctx.import_directory, ctx.do_reset)

if __name__ == '__main__':
    main()
