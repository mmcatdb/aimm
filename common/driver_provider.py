from typing import TypeVar
from common.config import Config
from common.drivers import MongoDriver, Neo4jDriver, PostgresDriver, PostgresConfig, MongoConfig, Neo4jConfig

TDriver = TypeVar('TDriver', PostgresDriver, MongoDriver, Neo4jDriver)

class DriverProvider:
    def __init__(self, db_configs: dict[str, PostgresConfig | MongoConfig | Neo4jConfig]):
        self.dbs = { id: DriverProvider._create_db(config) for id, config in db_configs.items() }
        self.all_ids = list(self.dbs.keys())

    @staticmethod
    def _create_db(config: PostgresConfig | MongoConfig | Neo4jConfig) -> PostgresDriver | MongoDriver | Neo4jDriver:
        if isinstance(config, PostgresConfig):
            return PostgresDriver(config)
        elif isinstance(config, MongoConfig):
            return MongoDriver(config)
        elif isinstance(config, Neo4jConfig):
            return Neo4jDriver(config)

    def get(self, id: str) -> PostgresDriver | MongoDriver | Neo4jDriver:
        db = self.dbs.get(id)
        if not db:
            raise ValueError(f'Database with id "{id}" not found in DatabaseProvider.')
        return db

    def get_typed(self, id: str, clazz: type[TDriver]) -> TDriver:
        db = self.get(id)
        if not isinstance(db, clazz):
            raise TypeError(f'Database with id "{id}" is not of type {clazz.__name__}.')
        return db

    @staticmethod
    def default(config: Config) -> 'DriverProvider':
        """
        Default providers for development and testing purposes.
        """
        return DriverProvider({
            'postgres': config.postgres,
            'mongo': config.mongo,
            'neo4j': config.neo4j,
        })

    def close(self):
        for db in self.dbs.values():
            db.close()
