from typing import TypeVar
from common.config import Config
from common.databases import Mongo, Neo4j, Postgres, PostgresConfig, MongoConfig, Neo4jConfig

TDatabase = TypeVar('TDatabase', Postgres, Mongo, Neo4j)

class DatabaseProvider:
    def __init__(self, db_configs: dict[str, PostgresConfig | MongoConfig | Neo4jConfig]):
        self.dbs = { id: DatabaseProvider._create_db(id, config) for id, config in db_configs.items() }
        self.all_ids = list(self.dbs.keys())

    @staticmethod
    def _create_db(id: str, config: PostgresConfig | MongoConfig | Neo4jConfig):
        if isinstance(config, PostgresConfig):
            return Postgres(config)
        elif isinstance(config, MongoConfig):
            return Mongo(config)
        elif isinstance(config, Neo4jConfig):
            return Neo4j(config)
        else:
            raise ValueError(f'Unsupported database config type for id "{id}".')

    def get(self, id: str) -> Postgres | Mongo | Neo4j:
        db = self.dbs.get(id)
        if not db:
            raise ValueError(f'Database with id "{id}" not found in DatabaseProvider.')
        return db

    def get_typed(self, id: str, clazz: type[TDatabase]) -> TDatabase:
        db = self.get(id)
        if not isinstance(db, clazz):
            raise TypeError(f'Database with id "{id}" is not of type {clazz.__name__}.')
        return db

    @staticmethod
    def default(config: Config):
        """
        Default providers for development and testing purposes.
        """
        return DatabaseProvider({
            'postgres': config.postgres,
            'mongo': config.mongo,
            'neo4j': config.neo4j,
        })
