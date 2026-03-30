import argparse
import json
import os
from common.config import Config, DatasetName
from common.driver_provider import DriverProvider
from common.drivers import DriverType, PostgresDriver, MongoDriver, Neo4jDriver
from common.utils import JsonEncoder, auto_close
from common.database import DatabaseInfo
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
    ctx.setup(type, args)

    if type == DriverType.POSTGRES:
        times = populate_postgres(ctx)
    elif type == DriverType.MONGO:
        times = populate_mongo(ctx)
    elif type == DriverType.NEO4J:
        times = populate_neo4j(ctx)
    else:
        raise ValueError(f'Unsupported database type: {type}')

    save_populate_times(ctx, times)

def common_args(parser: argparse.ArgumentParser):
    parser.add_argument('dataset', nargs=1, choices=get_available_dataset_names(), help=f'Name of the dataset.')
    parser.add_argument('--import-dir', type=str, default=None, help='Path to the directory containing the input files. If not specified, defaults to "{IMPORT_DIRECTORY}/<dataset>".')
    parser.add_argument('--no-reset', action='store_true', help='Skip clearing the database beforehand.')
    # FIXME Not ideal - make optional and specify the dataset elsewhere
    parser.add_argument('--scale', type=float, required=True, help='Scale factor of the generated dataset.')

class Context:
    def setup(self, type: DriverType, args: argparse.Namespace):
        self.dataset = DatasetName(args.dataset[0])
        self.database = DatabaseInfo(self.dataset, type, args.scale)  # type: ignore
        self.config = Config.load()
        self.dbs = DriverProvider.default(self.config)
        self.do_reset = not args.no_reset
        self.import_directory = args.import_dir or self.config.dataset_import_directory(self.dataset)

    def database_name(self) -> str:
        return self.database.dataset.value

def populate_postgres(ctx: Context):
    from common.loaders.postgres_loader import create_database_if_not_exists

    # Postgres databases have to be created explicitly, so let's do that.
    rootDriver = ctx.dbs.get(ctx.config.postgres.root_database, PostgresDriver)
    with auto_close(rootDriver):
        create_database_if_not_exists(rootDriver, ctx.database_name())

    driver = ctx.dbs.get(ctx.database_name(), PostgresDriver)
    with auto_close(driver):
        if ctx.dataset == DatasetName.EDBT:
            from datasets.edbt.postgres_loader import EdbtPostgresLoader
            loader = EdbtPostgresLoader(driver)
        elif ctx.dataset == DatasetName.TPCH:
            from datasets.tpch.postgres_loader import TpchPostgresLoader
            loader = TpchPostgresLoader(driver)
        else:
            raise ValueError(f'Unsupported dataset: {ctx.dataset}')

        return loader.run(ctx.import_directory, ctx.do_reset)

def populate_mongo(ctx: Context):
    # Mongo databases are created automatically by writing in them. So, nothing to do here.

    driver = ctx.dbs.get(ctx.database_name(), MongoDriver)
    with auto_close(driver):
        if ctx.dataset == DatasetName.EDBT:
            from datasets.edbt.mongo_loader import EdbtMongoLoader
            loader = EdbtMongoLoader(driver)
        elif ctx.dataset == DatasetName.TPCH:
            from datasets.tpch.mongo_loader import TpchMongoLoader
            loader = TpchMongoLoader(driver)
        else:
            raise ValueError(f'Unsupported dataset: {ctx.dataset}')

        return loader.run(ctx.import_directory, ctx.do_reset)

def populate_neo4j(ctx: Context):
    # Neo4j databases have to run in separate containers (the free edition doesn't support multiple databases). So, nothing to do here.

    driver = ctx.dbs.get(ctx.database_name(), Neo4jDriver)
    with auto_close(driver):
        if ctx.dataset == DatasetName.EDBT:
            from datasets.edbt.neo4j_loader import EdbtNeo4jLoader
            loader = EdbtNeo4jLoader(driver)
        elif ctx.dataset == DatasetName.TPCH:
            from datasets.tpch.neo4j_loader import TpchNeo4jLoader
            loader = TpchNeo4jLoader(driver)
        else:
            raise ValueError(f'Unsupported dataset: {ctx.dataset}')

        return loader.run(ctx.import_directory, ctx.do_reset)

def save_populate_times(ctx: Context, times: dict[str, float]):
    filename = f'{ctx.database.id()}.json'
    path = os.path.join(ctx.config.populate_directory, filename)
    with open(path, 'w') as file:
        json.dump(times, file, cls=JsonEncoder, indent=4)

def load_populate_times(config: Config, info: DatabaseInfo) -> dict[str, float]:
    filename = f'{info.id()}.json'
    path = os.path.join(config.populate_directory, filename)
    if not os.path.isfile(path):
        raise Exception(f'Populate times file not found: {path}')

    with open(path, 'r') as file:
        return json.load(file)

if __name__ == '__main__':
    main()
