from common.config import DatasetName
from common.database import DatabaseInfo, Database
from common.drivers import DriverType

# FIXME Use some unified database provider

DB_CACHE = dict[str, Database]()

def find_database(info: DatabaseInfo) -> Database:
    database = DB_CACHE.get(info.id())
    if not database:
        database = _find_database_uncached(info.dataset, info.type, info.scale)
        DB_CACHE[info.id()] = database

    return database

def _find_database_uncached(dataset: DatasetName, type: DriverType, scale: float | None) -> Database:
    if dataset == DatasetName.EDBT:
        if scale is None:
            raise ValueError('Scale factor must be provided for EDBT dataset.')

        if type == DriverType.POSTGRES:
            from datasets.edbt.postgres_database import EdbtPostgresDatabase
            return EdbtPostgresDatabase(scale)
        if type == DriverType.MONGO:
            from datasets.edbt.mongo_database import EdbtMongoDatabase
            return EdbtMongoDatabase(scale)
        if type == DriverType.NEO4J:
            from datasets.edbt.neo4j_database import EdbtNeo4jDatabase
            return EdbtNeo4jDatabase(scale)
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
