from common.config import DatasetName
from common.database import DatabaseInfo, Database
from common.drivers import DriverType

# FIXME Use some unified database provider

DB_CACHE: dict[str, Database] = {}

def find_database(info: DatabaseInfo) -> Database:
    database = DB_CACHE.get(info.id())
    if not database:
        database = find_database_uncached(info.dataset, info.type)
        DB_CACHE[info.id()] = database

    return database

def find_database_uncached(dataset: DatasetName, type: DriverType) -> Database:
    if dataset == DatasetName.EDBT:
        if type == DriverType.POSTGRES:
            from datasets.edbt.postgres_database import EdbtPostgresDatabase
            return EdbtPostgresDatabase()
        if type == DriverType.NEO4J:
            from datasets.edbt.neo4j_database import EdbtNeo4jDatabase
            return EdbtNeo4jDatabase()
    elif dataset == DatasetName.TPCH:
        if type == DriverType.POSTGRES:
            from datasets.tpch.postgres_database import TpchPostgresDatabase
            return TpchPostgresDatabase()
        if type == DriverType.MONGO:
            from datasets.tpch.mongo_database import TpchMongoDatabase
            return TpchMongoDatabase()
        if type == DriverType.NEO4J:
            from datasets.tpch.neo4j_database import TpchNeo4jDatabase
            return TpchNeo4jDatabase()

    raise ValueError(f'{type.value} is not supported for {dataset.value} dataset.')

def get_available_dataset_names() -> list[str]:
    return [dataset.value for dataset in DatasetName]

# FIXME this
TRAIN_DATASET = DatasetName.TPCH
