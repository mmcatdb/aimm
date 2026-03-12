from typing_extensions import deprecated
from common.database import Database
from common.drivers import DriverType
from common.driver_provider import DatasetName

@deprecated('Use some unified database provider')
def find_database(dataset: DatasetName, type: DriverType) -> Database:
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
        if type == DriverType.NEO4J:
            from datasets.tpch.neo4j_database import TpchNeo4jDatabase
            return TpchNeo4jDatabase()

    raise ValueError(f'{type.value} is not supported for {dataset.value} dataset.')

def get_available_dataset_names() -> list[str]:
    return [dataset.value for dataset in DatasetName]
