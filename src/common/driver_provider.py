from typing import TypeVar
from common.config import Config
from common.drivers import DriverType, MongoDriver, Neo4jDriver, PostgresDriver, PostgresConfig, MongoConfig, Neo4jConfig

TDriver = TypeVar('TDriver', PostgresDriver, MongoDriver, Neo4jDriver)
Driver = PostgresDriver | MongoDriver | Neo4jDriver

class DriverProvider:
    def __init__(self, db_configs: dict[DriverType, PostgresConfig | MongoConfig | Neo4jConfig]):
        self._drivers: dict[str, Driver] = {}
        self._db_configs = db_configs

    @staticmethod
    def default(config: Config) -> 'DriverProvider':
        """Default providers for development and testing purposes."""
        return DriverProvider({
            DriverType.POSTGRES : config.postgres,
            DriverType.MONGO: config.mongo,
            DriverType.NEO4J: config.neo4j,
        })

    def get(self, database: str, clazz: type[TDriver]) -> TDriver:
        """Use this to get a driver for *any* database, not just one of the predefined datasets."""
        type = self.__get_driver_type(clazz)
        key = self.__compute_key(type, database)
        driver = self._drivers.get(key)
        if not driver:
            config = self._db_configs.get(type)
            if not config:
                raise ValueError(f'Database config with type "{type.value}" not found in DriverProvider.')

            driver = self.__create_driver(config, database)
            self._drivers[key] = driver

        if not isinstance(driver, clazz):
            raise TypeError(f'Database driver with type "{type.value}" is not of type {clazz.__name__}.')

        return driver

    @staticmethod
    def __compute_key(type: DriverType, database: str) -> str:
        return f'{type.value}_{database}'

    @staticmethod
    def __get_driver_type(clazz: type[TDriver]) -> DriverType:
        if clazz == PostgresDriver:
            return DriverType.POSTGRES
        elif clazz == MongoDriver:
            return DriverType.MONGO
        elif clazz == Neo4jDriver:
            return DriverType.NEO4J
        else:
            raise ValueError(f'Unsupported driver class: {clazz}')

    @staticmethod
    def __create_driver(config: PostgresConfig | MongoConfig | Neo4jConfig, database: str) -> PostgresDriver | MongoDriver | Neo4jDriver:
        if isinstance(config, PostgresConfig):
            return PostgresDriver(config, database)
        elif isinstance(config, MongoConfig):
            return MongoDriver(config, database)
        elif isinstance(config, Neo4jConfig):
            return Neo4jDriver(config, database)
