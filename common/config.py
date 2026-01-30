from typing import Any
from typing_extensions import Self
import os
from pathlib import Path
from dotenv import load_dotenv
from .databases import PostgresConfig, MongoConfig, Neo4jConfig

# Load .env manually if needed (outside Docker)
defaultFilename = Path(__file__).resolve().parents[1] / '.env'

class Config:
    def __init__(self, postgres: PostgresConfig, mongo: MongoConfig, neo4j: Neo4jConfig, rest: dict[str, Any] = {}):
        self.postgres = postgres
        self.mongo = mongo
        self.neo4j = neo4j

        self.importDirectory = rest['import_directory']

    @staticmethod
    def load(filename: str | None = None) -> Self:
        try:
            load_dotenv(filename or defaultFilename)
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
            host = os.getenv('POSTGRES_HOST'),
            port = int(os.getenv('POSTGRES_PORT')),
            database = os.getenv('POSTGRES_DATABASE'),
            user = os.getenv('POSTGRES_USER'),
            password = os.getenv('POSTGRES_PASSWORD')
        )

    @staticmethod
    def loadMongo() -> MongoConfig:
        return MongoConfig(
            host = os.getenv('MONGO_HOST'),
            port = int(os.getenv('MONGO_PORT')),
            database = os.getenv('MONGO_DATABASE')
        )

    @staticmethod
    def loadNeo4j() -> Neo4jConfig:
        return Neo4jConfig(
            host = os.getenv('NEO4J_HOST'),
            port = int(os.getenv('NEO4J_PORT')),
            user = os.getenv('NEO4J_USER'),
            password = os.getenv('NEO4J_PASSWORD')
        )

    @staticmethod
    def loadRest() -> dict[str, Any]:
        return {
            'import_directory': os.getenv('IMPORT_DIRECTORY')
        }
