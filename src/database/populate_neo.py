from common.config import Config
from common.drivers import Neo4jDriver
from datasets.tpch.neo4j_loader import TpchNeo4jLoader

def main():
    config = Config.load()

    driver = Neo4jDriver(config.neo4j)

    loader = TpchNeo4jLoader(config, driver)

    loader.run()

if __name__ == '__main__':
    main()
