import argparse
import os
import random
import numpy as np
from core.config import Config
from core.drivers import DriverType
from core.query import QueryMeasurement, MeasuredQueries, TQuery, load_measured, parse_query_instance_id
from core.utils import exit_with_error, exit_with_exception, print_warning
from latency_estimation.class_provider import BaseFeatureExtractor, get_feature_extractor
from latency_estimation.dataset import ArrayDataset, DatasetId, DatasetItem, create_dataset_id, parse_dataset_id, save_dataset, try_save_available_operators
from latency_estimation.feature_extractor import load_feature_extractor, save_feature_extractor
from latency_estimation.model import OperatorCollector
from providers.path_provider import PathProvider

# TODO argument? config? some more robust function?
SKIP_FIRST_MEASURED_LATENCIES = 5

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Combine query measurements from multiple databases into a single dataset and a feature extractor, optimized for ML training.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()

    try:
        run(config, args)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('dataset_id', nargs=1,   help='Id of the output dataset. Pattern: {driver_type}/{dataset_name}.')
    # TODO Allow some "all" option, or read the input from a file or sth.
    parser.add_argument('measurement_suffixes', nargs='+', help='Paths to the input measurement files. Pattern: {schema_id}/measured-{num_queries}-{num_runs}.jsonl')
    parser.add_argument('--feature-extractor-dataset', type=str, help='Name of the dataset whose feature extractor should be used (instead of creating a new one). Used to ensure consistent vocabularies across train, validation, and test datasets.')

    parser.add_argument('--val-dataset', type=str, help='Optional validation dataset id. Pattern: {driver_type}/{dataset_name}.')
    parser.add_argument('--val-ratio',   type=float, default=0.2, help='Fraction of measurements to put into --val-dataset.')
    parser.add_argument('--split-seed',  type=int, default=69, help='Random seed used for train/validation splitting.')

def run(config: Config, args: argparse.Namespace):
    pp = PathProvider(config)

    dataset_id = args.dataset_id[0]
    fe_dataset = args.feature_extractor_dataset
    val_dataset_id = args.val_dataset

    driver_type, dataset = parse_dataset_id(dataset_id)
    if val_dataset_id is not None:
        val_driver_type, _ = parse_dataset_id(val_dataset_id)
        if val_driver_type != driver_type:
            raise ValueError(f'Validation dataset driver "{val_driver_type.value}" must match output driver "{driver_type.value}".')
        if not 0 < args.val_ratio < 1:
            raise ValueError('--val-ratio must be between 0 and 1.')

    if os.path.exists(pp.dataset(dataset_id)):
        print_warning(f'Dataset with id "{dataset_id}" already exists. It will be overwritten.')
    if os.path.exists(pp.feature_extractor(dataset_id)):
        print_warning(f'Feature extractor for dataset id "{dataset_id}" already exists. It will be overwritten.')
    if val_dataset_id is not None:
        if os.path.exists(pp.dataset(val_dataset_id)):
            print_warning(f'Dataset with id "{val_dataset_id}" already exists. It will be overwritten.')
        if os.path.exists(pp.feature_extractor(val_dataset_id)):
            print_warning(f'Feature extractor for dataset id "{val_dataset_id}" already exists. It will be overwritten.')

    all_measured = _load_all_measured(pp, driver_type, args.measurement_suffixes)

    if fe_dataset:
        # Use an existing feature extractor - it should not be changed.
        fe_dataset_id = create_dataset_id(driver_type, fe_dataset)
        feature_extractor = load_feature_extractor(pp.feature_extractor(fe_dataset_id))
    else:
        # Create a new feature extractor from plans.
        feature_extractor = get_feature_extractor(driver_type)
        for measured in all_measured:
            plans = [item.plan for item in measured.items]
            feature_extractor.extend_vocabularies(plans, measured.global_stats)

    dataset_items = list[DatasetItem]()

    for measured in all_measured:
        feature_extractor.set_global_stats(measured.global_stats)
        for measurement in measured.items:
            dataset_items.append(create_dataset_item(feature_extractor, measurement))

    if val_dataset_id is None:
        _save_dataset_artifacts(pp, dataset_id, dataset_items, feature_extractor)
        return

    train_items, val_items = split_dataset_items(dataset_items, args.val_ratio, args.split_seed)
    print(f'Split {len(dataset_items)} items into {len(train_items)} train and {len(val_items)} validation items.')
    _save_dataset_artifacts(pp, dataset_id, train_items, feature_extractor)
    _save_dataset_artifacts(pp, val_dataset_id, val_items, feature_extractor)

def _save_dataset_artifacts(pp: PathProvider, dataset_id: DatasetId, dataset_items: list[DatasetItem], feature_extractor: BaseFeatureExtractor):
    dataset = ArrayDataset(dataset_items)
    save_dataset(pp.dataset(dataset_id), dataset)
    save_feature_extractor(pp.feature_extractor(dataset_id), feature_extractor)

    operators = OperatorCollector.run([item.plan for item in dataset_items])
    try_save_available_operators(pp.operators(dataset_id), operators)

def _load_all_measured(pp: PathProvider, driver_type: DriverType, measurement_suffixes: list[str]) -> list[MeasuredQueries[TQuery]]:
    all = list[MeasuredQueries[TQuery]]()
    for suffix in measurement_suffixes:
        path = pp.measured_by_suffix(driver_type, suffix)
        try:
            all.append(load_measured(path))
        except Exception as e:
            print_warning(f'Error processing measurements file "{path}". Skipping.', e)

    if not all:
        exit_with_error('No valid measurements files found for driver type "{driver_type}" and suffixes {measurement_suffixes}.')

    return all

def create_dataset_item(feature_extractor: BaseFeatureExtractor, measurement: QueryMeasurement[TQuery]) -> DatasetItem:
    plan = feature_extractor.extract_plan(measurement.plan)
    structure_hash = feature_extractor.compute_plan_structure_hash(plan)
    # TODO Do some statistics about the measured latencies, filter outliers, etc.
    trimmed_times = measurement.times[SKIP_FIRST_MEASURED_LATENCIES:]
    if not trimmed_times:
        trimmed_times = measurement.times
    latency = np.mean(trimmed_times).item()

    return DatasetItem(measurement.id, measurement.label, latency, plan, structure_hash)

def split_dataset_items(dataset_items: list[DatasetItem], val_ratio: float, seed: int) -> tuple[list[DatasetItem], list[DatasetItem]]:
    rng = random.Random(seed)
    groups = dict[str, list[DatasetItem]]()
    for item in dataset_items:
        _, template_name, _ = parse_query_instance_id(item.query_id)
        groups.setdefault(template_name, []).append(item)

    train_items = list[DatasetItem]()
    val_items = list[DatasetItem]()

    for items in groups.values():
        shuffled = items[:]
        rng.shuffle(shuffled)

        if len(shuffled) < 2:
            train_items.extend(shuffled)
            continue

        val_count = max(1, round(len(shuffled) * val_ratio))
        val_count = min(val_count, len(shuffled) - 1)
        val_items.extend(shuffled[:val_count])
        train_items.extend(shuffled[val_count:])

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items

if __name__ == '__main__':
    main()
