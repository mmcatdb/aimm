import argparse
import os
import numpy as np
from core.config import Config
from core.query import QueryMeasurement, TQuery, load_measured, print_warning
from core.utils import exit_with_exception
from latency_estimation.class_provider import BaseFeatureExtractor, get_feature_extractor
from latency_estimation.dataset import ArrayDataset, DatasetId, DatasetItem, parse_dataset_id, save_dataset, try_save_available_operators
from latency_estimation.feature_extractor import save_feature_extractor
from latency_estimation.model import OperatorCollector
from ..providers.path_provider import PathProvider
from .measure_queries import MeasuredQueries

# TODO argument? config? some more robust function?
SKIP_FIRST_MEASURED_LATENCIES = 5

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Combine query measurements from multiple databases into a single dataset and a feature extractor, optimized for ML training.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()
    dataset_id = args.dataset_id[0]

    try:
        run(config, dataset_id, args.measurement_paths)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('dataset_id', nargs=1,   help='Id of the output dataset. Pattern: {driver_type}/{dataset_name}.')
    # TODO Allow some "all" option, or read the input from a file or sth.
    parser.add_argument('measurement_paths', nargs='+', help='Paths to the input measurement files.')

def run(config: Config, dataset_id: DatasetId, measurement_paths: list[str]):
    pp = PathProvider(config)

    driver_type, dataset = parse_dataset_id(dataset_id)

    if os.path.exists(pp.dataset(dataset_id)):
        print_warning(f'Dataset with id "{dataset_id}" already exists.')
    if os.path.exists(pp.feature_extractor(dataset_id)):
        print_warning(f'Feature extractor for dataset id "{dataset_id}" already exists.')

    # FIXME Use feature extractor from another dataset - so that the vocabularies are consistent.

    feature_extractor = get_feature_extractor(driver_type)

    all_measured = _load_all_measured(measurement_paths)
    for measured in all_measured:
        plans = [item.plan for item in measured.items]
        feature_extractor.extend_vocabularies(plans, measured.global_stats)

    dataset_items = list[DatasetItem]()

    for measured in all_measured:
        for item in measured.items:
            dataset_items.append(create_dataset_item(feature_extractor, item))

    dataset = ArrayDataset(dataset_items)

    save_dataset(pp.dataset(dataset_id), dataset)
    save_feature_extractor(pp.feature_extractor(dataset_id), feature_extractor)

    operators = OperatorCollector.run([item.plan for item in dataset_items])
    try_save_available_operators(pp.operators(dataset_id), operators)

def _load_all_measured(paths: list[str]) -> list[MeasuredQueries[TQuery]]:
    all = list[MeasuredQueries[TQuery]]()
    for path in paths:
        try:
            all.append(load_measured(path))
        except Exception as e:
            print_warning(f'Error processing measurements file "{path}". Skipping.', e)

    return all

def create_dataset_item(feature_extractor: BaseFeatureExtractor, item: QueryMeasurement[TQuery]) -> DatasetItem:
    plan = feature_extractor.extract_plan(item.plan)
    structure_hash = feature_extractor.compute_plan_structure_hash(plan)
    # TODO Do some statistics about the measured latencies, filter outliers, etc.
    latency = np.mean(item.times[SKIP_FIRST_MEASURED_LATENCIES:]).item()

    return DatasetItem(item.query_id, latency, plan, structure_hash)

if __name__ == '__main__':
    main()
