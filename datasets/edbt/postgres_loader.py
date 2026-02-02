from typing_extensions import override
from common.config import Config
from common.drivers import PostgresDriver
from common.loaders.postgres_loader import PostgresLoader

class EdbtPostgresLoader(PostgresLoader):
    def __init__(self, config: Config, driver: PostgresDriver):
        super().__init__(config, driver)

    @override
    def name(self) -> str:
        return 'EDBT'

    @override
    def _get_schemas(self) -> dict[str, list[dict]]:
        # TODO
        return {}


