from common.config import Config
from common.drivers import PostgresDriver
from datasets.edbt.postgres_loader import EdbtPostgresLoader

def main():
    config = Config.load()

    driver = PostgresDriver(config.postgres)

    loader = EdbtPostgresLoader(config, driver)

    loader.run()

if __name__ == '__main__':
    main()
