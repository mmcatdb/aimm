from enum import Enum
from typing import Any
import os
from pathlib import Path
from dotenv import load_dotenv
from common.drivers import PostgresConfig, MongoConfig, Neo4jConfig
from common.utils import exit_with_error

# from datetime import datetime
# GLOBAL_RNG_SEED = datetime.now().timestamp()
GLOBAL_RNG_SEED = 69

class DatasetName(Enum):
    TPCH = 'tpch'
    EDBT = 'edbt'

    def label(self) -> str:
        """Get a human-readable label for this dataset."""
        if self == DatasetName.TPCH:
            return 'TPC-H'
        elif self == DatasetName.EDBT:
            return 'EDBT'
        else:
            return self.value

class Config:
    def __init__(self, postgres: PostgresConfig, mongo: MongoConfig, neo4j: Neo4jConfig, rest: dict[str, Any] = {}):
        self.postgres = postgres
        self.mongo = mongo
        self.neo4j = neo4j

        self.import_directory: str = rest['import_directory']
        self.cache_directory: str = rest['cache_directory']
        self.checkpoints_directory: str = rest['checkpoints_directory']
        self.results_directory: str = rest['results_directory']
        self.metrics_directory: str = rest['metrics_directory']
        self.populate_directory: str = rest['populate_directory']
        self.measure_directory: str = rest['measure_directory']
        self.experiments_directory: str = rest['experiments_directory']
        self.device: str = rest['device']
        self.train_num_runs: int | None = rest.get('train_num_runs')
        self.num_queries: int | None = rest.get('num_queries')
        self.num_epochs: int | None = rest.get('num_epochs')

    # Load .env manually if needed (outside Docker).
    DEFAULT_CONFIG_PATH = Path.joinpath(Path.cwd(), '.env')

    def dataset_import_directory(self, dataset: DatasetName) -> str:
        return os.path.join(self.import_directory, dataset.value)

    @staticmethod
    def load(path: str | None = None) -> 'Config':
        try:
            load_dotenv(path or Config.DEFAULT_CONFIG_PATH)
        except ImportError as e:
            exit_with_error('Can\'t load .env file.', e)

        return Config(
            Config.__loadPostgres(),
            Config.__loadMongo(),
            Config.__loadNeo4j(),
            Config.__loadRest(),
        )

    @staticmethod
    def __loadPostgres() -> PostgresConfig:
        return PostgresConfig(
            host = _string('POSTGRES_HOST'),
            port = _int('POSTGRES_PORT'),
            root_database = _string('POSTGRES_ROOT_DATABASE'),
            user = _string('POSTGRES_USER'),
            password = _string('POSTGRES_PASSWORD'),
        )

    @staticmethod
    def __loadMongo() -> MongoConfig:
        return MongoConfig(
            host = _string('MONGO_HOST'),
            port = _int('MONGO_PORT'),
        )

    @staticmethod
    def __loadNeo4j() -> Neo4jConfig:
        return Neo4jConfig(
            host = _string('NEO4J_HOST'),
            ports = {
                # TODO Use dataset names
                'tpch': _int('NEO4J_TPCH_PORT'),
                'edbt': _int('NEO4J_EDBT_PORT'),
            },
            user = _string('NEO4J_USER'),
            password = _string('NEO4J_PASSWORD'),
        )

    @staticmethod
    def __loadRest() -> dict[str, Any]:
        return {
            'import_directory': _string('IMPORT_DIRECTORY', 'data/inputs'),
            'cache_directory': _string('CACHE_DIRECTORY', 'data/cache'),
            'checkpoints_directory': _string('CHECKPOINTS_DIRECTORY', 'data/checkpoints'),
            'results_directory': _string('RESULTS_DIRECTORY', 'data'),
            'metrics_directory': _string('METRICS_DIRECTORY', 'data/metrics'),
            'populate_directory': _string('POPULATE_DIRECTORY', 'data/populate'),
            'measure_directory': _string('MEASURE_DIRECTORY', 'data/measure'),
            'experiments_directory': _string('EXPERIMENTS_DIRECTORY', 'data/experiments'),
            'device': _string('DEVICE'),
            'train_num_runs': _int_optional('TRAIN_NUM_RUNS'),
            'num_queries': _int_optional('NUM_QUERIES'),
            'num_epochs': _int_optional('NUM_EPOCHS'),
        }

def _string(key: str, default: str | None = None) -> str:
    value = os.getenv(key, default)
    if value is None or (value == '' and default != ''):
        raise ValueError(f'Environment variable "{key}" is not set.')
    return value

def _string_optional(key: str) -> str | None:
    value = os.getenv(key)
    return None if value == '' else value

def _int(key: str, default: str | None = None) -> int:
    return int(_string(key, default))

def _int_optional(key: str) -> int | None:
    value = _string_optional(key)
    return int(value) if value is not None else None
