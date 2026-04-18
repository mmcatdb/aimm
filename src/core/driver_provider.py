import re
from typing import TypeVar
from core.config import Config
from core.drivers import DriverType, MongoDriver, Neo4jDriver, PostgresDriver, PostgresConfig, MongoConfig, Neo4jConfig
from core.query import SchemaName, create_schema_id

TDriver = TypeVar('TDriver', PostgresDriver, MongoDriver, Neo4jDriver)
Driver = PostgresDriver | MongoDriver | Neo4jDriver

class DriverProvider:
    def __init__(self, db_configs: dict[DriverType, PostgresConfig | MongoConfig | Neo4jConfig]):
        self._drivers = dict[str, Driver]()
        self._db_configs = db_configs

    @staticmethod
    def default(config: Config) -> 'DriverProvider':
        """Default providers for development and testing purposes."""
        return DriverProvider({
            DriverType.POSTGRES: config.postgres,
            DriverType.MONGO: config.mongo,
            DriverType.NEO4J: config.neo4j,
        })

    # For some reason it's not trivial to get the class from the type and then pass it to a type[Driver] argument.
    # So we just just have several get methods for different use cases.

    def get(self, type: DriverType, schema: SchemaName, scale: float) -> Driver:
        database = self.compute_database_name(schema, scale)
        return self.__get_or_create(type, database)

    def get_typed(self, clazz: type[TDriver], schema: SchemaName, scale: float) -> TDriver:
        database = self.compute_database_name(schema, scale)
        return self.get_by_name(clazz, database)

    def get_by_name(self, clazz: type[TDriver], database: str) -> TDriver:
        """Use this to get a driver for *any* database, not just one of the predefined datasets."""
        type = self.__get_driver_type(clazz)
        driver = self.__get_or_create(type, database)

        if not isinstance(driver, clazz):
            raise TypeError(f'Database driver with type "{type.value}" is not of type {clazz.__name__}.')

        return driver

    def __get_or_create(self, type: DriverType, database: str) -> Driver:
        key = self.__compute_key(type, database)
        driver = self._drivers.get(key)
        if not driver:
            config = self._db_configs.get(type)
            if not config:
                raise ValueError(f'Database config with type "{type.value}" not found in DriverProvider.')

            driver = self.__create_driver(config, database)
            self._drivers[key] = driver

        return driver

    DB_NAME_REPLACE_REGEX = re.compile(r'[^\w]')

    @staticmethod
    def compute_database_name(schema: SchemaName, scale: float) -> str:
        """Produces a valid database name for the specific driver type.

        For example, Postgres doesn't allow dots in database names, so we replace them with underscores.
        """

        # The rules for postgres and mongo should be [a-zA-Z_][a-zA-Z0-9_]{0,63}.
        # Neo4j has something similar but we just can't care less because we have to use the default database for it. But let's just use the similar rules, they seem to be docker-friendly as well.
        # The schema name should be valid already, but just in case ...

        return DriverProvider.DB_NAME_REPLACE_REGEX.sub('_', create_schema_id(schema, scale))

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

