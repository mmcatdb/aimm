from enum import Enum
from typing import Any
import os
from pathlib import Path
from dotenv import load_dotenv
from common.drivers import PostgresConfig, MongoConfig, Neo4jConfig
from common.utils import exit_with_error

class DatasetName(Enum):
    TPCH = 'tpch'
    EDBT = 'edbt'

class Config:
    def __init__(self, postgres: PostgresConfig, mongo: MongoConfig, neo4j: Neo4jConfig, rest: dict[str, Any] = {}):
        self.postgres = postgres
        self.mongo = mongo
        self.neo4j = neo4j

        self.import_directory: str = rest['import_directory']
        self.cache_directory: str = rest['cache_directory']
        self.checkpoint_directory: str = rest['checkpoint_directory']
        self.results_directory: str = rest['results_directory']
        self.device: str = rest['device']

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
            'checkpoint_directory': _string('CHECKPOINT_DIRECTORY', 'data/checkpoints'),
            'results_directory': _string('RESULTS_DIRECTORY', 'data'),
            'device': _string('DEVICE'),
        }

def _string(key: str, default: str | None = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f'Environment variable "{key}" is not set.')
    return value

def _string_optional(key: str) -> str | None:
    return os.getenv(key)

def _int(key: str, default: str | None = None) -> int:
    return int(_string(key, default))

def _int_optional(key: str) -> int | None:
    value = os.getenv(key)
    return int(value) if value is not None else None
