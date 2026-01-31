from typing import Any
from typing_extensions import Self
import os
from pathlib import Path
from dotenv import load_dotenv
from .drivers import PostgresConfig, MongoConfig, Neo4jConfig

def _string(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise ValueError(f'Environment variable "{key}" is not set.')
    return value

def _int(key: str) -> int:
    return int(_string(key))

class Config:
    def __init__(self, postgres: PostgresConfig, mongo: MongoConfig, neo4j: Neo4jConfig, rest: dict[str, Any] = {}):
        self.postgres = postgres
        self.mongo = mongo
        self.neo4j = neo4j

        self.import_directory: str = rest['import_directory']

    # Load .env manually if needed (outside Docker).
    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / '.env'

    @staticmethod
    def load(path: str | None = None) -> 'Config':
        try:
            load_dotenv(path or Config.DEFAULT_CONFIG_PATH)
        except ImportError:
            print('Error: can\'t load .env file')
            exit(1)

        return Config(
            Config.loadPostgres(),
            Config.loadMongo(),
            Config.loadNeo4j(),
            Config.loadRest(),
        )

    @staticmethod
    def loadPostgres() -> PostgresConfig:
        return PostgresConfig(
            host = _string('POSTGRES_HOST'),
            port = _int('POSTGRES_PORT'),
            database = _string('POSTGRES_DATABASE'),
            user = _string('POSTGRES_USER'),
            password = _string('POSTGRES_PASSWORD')
        )

    @staticmethod
    def loadMongo() -> MongoConfig:
        return MongoConfig(
            host = _string('MONGO_HOST'),
            port = _int('MONGO_PORT'),
            database = _string('MONGO_DATABASE')
        )

    @staticmethod
    def loadNeo4j() -> Neo4jConfig:
        return Neo4jConfig(
            host = _string('NEO4J_HOST'),
            port = _int('NEO4J_PORT'),
            user = _string('NEO4J_USER'),
            password = _string('NEO4J_PASSWORD')
        )

    @staticmethod
    def loadRest() -> dict[str, Any]:
        return {
            'import_directory': _string('IMPORT_DIRECTORY')
        }

    @staticmethod
    def _string(key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f'Environment variable "{key}" is not set.')
        return value
