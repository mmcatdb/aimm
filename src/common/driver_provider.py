from enum import Enum
import os
from typing import TypeVar
from common.config import Config
from common.drivers import DriverType, MongoDriver, Neo4jDriver, PostgresDriver, PostgresConfig, MongoConfig, Neo4jConfig

class DatasetName(Enum):
    TPCH = 'tpch'
    EDBT = 'edbt'

# TODO This should not be here.
def dataset_import_directory(config: Config, dataset: DatasetName) -> str:
    return os.path.join(config.import_directory, dataset.value)

TDriver = TypeVar('TDriver', PostgresDriver, MongoDriver, Neo4jDriver)
Driver = PostgresDriver | MongoDriver | Neo4jDriver

class DriverProvider:
    def __init__(self, db_configs: dict[DriverType, PostgresConfig | MongoConfig | Neo4jConfig]):
        self._drivers: dict[str, Driver] = {}
        self._db_configs = db_configs
        self._available_drivers = [driver.value for driver in DriverType]

    def get(self, dataset: DatasetName, clazz: type[TDriver]) -> TDriver:
        """Convenience method for predefined datasets."""
        return self.get_for_database(dataset.value, clazz)

    def get_for_database(self, database: str, clazz: type[TDriver]) -> TDriver:
        """Use this to get a driver for *any* database, not just one of the predefined datasets."""
        type = self._get_driver_type(clazz)
        key = self._compute_key(type, database)
        driver = self._drivers.get(key)
        if not driver:
            config = self._db_configs.get(type)
            if not config:
                raise ValueError(f'Database config with type "{type.value}" not found in DriverProvider.')

            driver = self._create_db(config, database)
            self._drivers[key] = driver

        if not isinstance(driver, clazz):
            raise TypeError(f'Database driver with type "{type.value}" is not of type {clazz.__name__}.')

        return driver

    # TODO Not ideal but works for now.
    def available_drivers(self) -> list[str]:
        return self._available_drivers

    @staticmethod
    def _compute_key(type: DriverType, database: str) -> str:
        return f'{type.value}_{database}'

    @staticmethod
    def _get_driver_type(clazz: type[TDriver]) -> DriverType:
        if clazz == PostgresDriver:
            return DriverType.POSTGRES
        elif clazz == MongoDriver:
            return DriverType.MONGO
        elif clazz == Neo4jDriver:
            return DriverType.NEO4J
        else:
            raise ValueError(f'Unsupported driver class: {clazz}')

    @staticmethod
    def _create_db(config: PostgresConfig | MongoConfig | Neo4jConfig, database: str) -> PostgresDriver | MongoDriver | Neo4jDriver:
        if isinstance(config, PostgresConfig):
            return PostgresDriver(config, database)
        elif isinstance(config, MongoConfig):
            return MongoDriver(config)
        elif isinstance(config, Neo4jConfig):
            return Neo4jDriver(config, database)

    @staticmethod
    def default(config: Config) -> 'DriverProvider':
        """Default providers for development and testing purposes."""
        return DriverProvider({
            DriverType.POSTGRES : config.postgres,
            DriverType.MONGO: config.mongo,
            DriverType.NEO4J: config.neo4j,
        })

    def close(self):
        for driver in self._drivers.values():
            driver.close()
