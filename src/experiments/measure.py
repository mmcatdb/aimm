import os
import json
import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
from tabulate import tabulate
from common.config import Config, DatasetName
from common.database import Database, DatabaseInfo
from common.drivers import DriverType, PostgresDriver, MongoDriver, Neo4jDriver
from common.query_registry import TQuery
from datasets.databases import find_database
from latency_estimation.plan_extractor import BasePlanExtractor

DATASET = DatasetName.EDBT
NUM_RUNS = 100
DRY_RUN = True

def main():
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--scale', type=float, required=True, help='Scale factor of the generated dataset.')
    args = parser.parse_args()

    scale = args.scale
    config = Config.load()

    process_postgres(config, scale)
    process_mongo(config, scale)
    process_neo4j(config, scale)

def process_postgres(config: Config, scale: float):
    from latency_estimation.postgres.plan_extractor import PlanExtractor
    database = find_database(DatabaseInfo(DATASET, DriverType.POSTGRES, scale))
    driver = PostgresDriver(config.postgres, DATASET.value)
    plan_extractor = PlanExtractor(driver)
    process_database(database, plan_extractor)

def process_mongo(config: Config, scale: float):
    from latency_estimation.mongo.plan_extractor import PlanExtractor
    database = find_database(DatabaseInfo(DATASET, DriverType.MONGO, scale))
    driver = MongoDriver(config.mongo, DATASET.value)
    plan_extractor = PlanExtractor(driver)
    process_database(database, plan_extractor)

def process_neo4j(config: Config, scale: float):
    from latency_estimation.neo4j.plan_extractor import PlanExtractor
    database = find_database(DatabaseInfo(DATASET, DriverType.NEO4J, scale))
    driver = Neo4jDriver(config.neo4j, DATASET.value)
    plan_extractor = PlanExtractor(driver)
    process_database(database, plan_extractor)

def process_database(database: Database[TQuery], plan_extractor: BasePlanExtractor[TQuery]):
    results = measure_database(database, plan_extractor)
    print_database(database, results)
    if not DRY_RUN:
        save_database_measurement(Config.load(), database, results)

@dataclass
class QueryResult:
    id: str
    times: list[float]
    mean: float
    std: float

    @staticmethod
    def from_dict(d: dict) -> 'QueryResult':
        return QueryResult(
            id=d['id'],
            times=d['times'],
            mean=d['mean'],
            std=d['std'],
        )

def measure_database(database: Database[TQuery], plan_extractor: BasePlanExtractor[TQuery]):
    output = list[QueryResult]()

    for def_ in database.get_query_defs():
        times = plan_extractor.measure_query_generated(def_, NUM_RUNS)
        result = QueryResult(
            id=def_.id,
            times=times,
            mean=np.mean(times).item(),
            std=np.std(times).item(),
        )
        output.append(result)

    return output

def print_database(info: DatabaseInfo, results: Sequence[QueryResult]):
    table = []
    for item in results:
        row = [
            item.id,
            f'{item.mean:.2f}',
            f'{item.std:.2f}',
            f'{item.std / item.mean * 100:.2f}' if item.mean != 0 else '0.00',
        ]

        table.append(row)

    headers = [
        'ID',
        'Mean [ms]',
        'STD [ms]',
        'RSD [%]',
    ]

    print(info.label())
    print(tabulate(table, headers=headers, tablefmt='grid'))

def save_database_measurement(config: Config, info: DatabaseInfo, results: Sequence[QueryResult]):
    filename = f'{info.id()}.json'
    path = os.path.join(config.measure_directory, filename)
    with open(path, 'w') as file:
        json.dump([result.__dict__ for result in results], file, indent=4)

def load_database_measurement(config: Config, info: DatabaseInfo) -> list[QueryResult]:
    filename = f'{info.id()}.json'
    path = os.path.join(config.measure_directory, filename)
    with open(path, 'r') as file:
        data = json.load(file)
        return [QueryResult.from_dict(item) for item in data]

if __name__ == '__main__':
    main()
