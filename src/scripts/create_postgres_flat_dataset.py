import argparse
import os
import random

import numpy as np

from core.config import Config
from core.drivers import DriverType
from core.query import QueryMeasurement, TQuery, load_measured, parse_query_instance_id
from core.utils import exit_with_error, exit_with_exception, print_warning
from latency_estimation.dataset import DatasetId, create_dataset_id, parse_dataset_id
from latency_estimation.postgres.flat_dataset import FlatArrayDataset, FlatDatasetItem, save_flat_dataset
from latency_estimation.postgres.flat_feature_extractor import (
    FlatFeatureExtractor,
    load_flat_feature_extractor,
    save_flat_feature_extractor,
)
from providers.path_provider import PathProvider


SKIP_FIRST_MEASURED_LATENCIES = 5


def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Create a PostgreSQL flat-feature dataset from measured query JSONL files.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    try:
        run(Config.load(), args)
    except Exception as e:
        exit_with_exception(e)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('dataset_id', nargs=1, help='Id of the output dataset. Pattern: postgres/{dataset_name}.')
    parser.add_argument('measurement_suffixes', nargs='+', help='Paths to measurement files. Pattern: {schema_id}/measured-{num_queries}-{num_runs}.jsonl')
    parser.add_argument('--feature-extractor-dataset', type=str, help='Dataset name whose flat feature extractor should be reused.')
    parser.add_argument('--val-dataset', type=str, help='Optional validation dataset id. Pattern: postgres/{dataset_name}.')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Fraction of measurements to put into --val-dataset.')
    parser.add_argument('--split-seed', type=int, default=69, help='Random seed used for train/validation splitting.')
    parser.add_argument('--include-schema-identifiers', action='store_true', help='Include Relation Name and Index Name features. Off by default for cross-schema evaluation.')


def run(config: Config, args: argparse.Namespace):
    pp = PathProvider(config)

    dataset_id = args.dataset_id[0]
    driver_type, _ = parse_dataset_id(dataset_id)
    _require_postgres(driver_type)

    val_dataset_id = args.val_dataset
    if val_dataset_id is not None:
        val_driver_type, _ = parse_dataset_id(val_dataset_id)
        _require_postgres(val_driver_type)
        if not 0 < args.val_ratio < 1:
            raise ValueError('--val-ratio must be between 0 and 1.')

    _warn_if_exists(pp.flat_dataset(dataset_id), f'Flat dataset with id "{dataset_id}"')
    _warn_if_exists(pp.flat_feature_extractor(dataset_id), f'Flat feature extractor for dataset id "{dataset_id}"')
    if val_dataset_id is not None:
        _warn_if_exists(pp.flat_dataset(val_dataset_id), f'Flat dataset with id "{val_dataset_id}"')
        _warn_if_exists(pp.flat_feature_extractor(val_dataset_id), f'Flat feature extractor for dataset id "{val_dataset_id}"')

    all_measured = _load_all_measured(pp, args.measurement_suffixes)

    if args.feature_extractor_dataset:
        fe_dataset_id = create_dataset_id(DriverType.POSTGRES, args.feature_extractor_dataset)
        feature_extractor = load_flat_feature_extractor(pp.flat_feature_extractor(fe_dataset_id))
    else:
        feature_extractor = FlatFeatureExtractor(include_schema_identifiers=args.include_schema_identifiers)
        feature_extractor.fit([measurement.plan for measured in all_measured for measurement in measured.items])

    items = list[FlatDatasetItem]()
    for measured in all_measured:
        for measurement in measured.items:
            items.append(create_flat_dataset_item(feature_extractor, measurement))

    if val_dataset_id is None:
        _save_flat_dataset_artifacts(pp, dataset_id, items, feature_extractor)
        return

    train_items, val_items = split_dataset_items(items, args.val_ratio, args.split_seed)
    print(f'Split {len(items)} items into {len(train_items)} train and {len(val_items)} validation items.')
    _save_flat_dataset_artifacts(pp, dataset_id, train_items, feature_extractor)
    _save_flat_dataset_artifacts(pp, val_dataset_id, val_items, feature_extractor)


def create_flat_dataset_item(feature_extractor: FlatFeatureExtractor, measurement: QueryMeasurement[TQuery]) -> FlatDatasetItem:
    trimmed_times = measurement.times[SKIP_FIRST_MEASURED_LATENCIES:]
    if not trimmed_times:
        trimmed_times = measurement.times
    latency = np.mean(trimmed_times).item()

    return FlatDatasetItem(
        query_id=measurement.id,
        label=measurement.label,
        latency=latency,
        features=feature_extractor.transform_plan(measurement.plan),
        plan=measurement.plan,
    )


def split_dataset_items(dataset_items: list[FlatDatasetItem], val_ratio: float, seed: int) -> tuple[list[FlatDatasetItem], list[FlatDatasetItem]]:
    rng = random.Random(seed)
    groups = dict[str, list[FlatDatasetItem]]()
    for item in dataset_items:
        _, template_name, _ = parse_query_instance_id(item.query_id)
        groups.setdefault(template_name, []).append(item)

    train_items = list[FlatDatasetItem]()
    val_items = list[FlatDatasetItem]()

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


def _save_flat_dataset_artifacts(pp: PathProvider, dataset_id: DatasetId, items: list[FlatDatasetItem], feature_extractor: FlatFeatureExtractor):
    save_flat_dataset(pp.flat_dataset(dataset_id), FlatArrayDataset(items))
    save_flat_feature_extractor(pp.flat_feature_extractor(dataset_id), feature_extractor)
    print(f'Saved {len(items)} flat items to {pp.flat_dataset(dataset_id)}')
    print(f'Feature dimension: {len(feature_extractor.feature_names)}')


def _load_all_measured(pp: PathProvider, measurement_suffixes: list[str]):
    all_measured = []
    for suffix in measurement_suffixes:
        path = pp.measured_by_suffix(DriverType.POSTGRES, suffix)
        try:
            all_measured.append(load_measured(path))
        except Exception as e:
            print_warning(f'Error processing measurements file "{path}". Skipping.', e)

    if not all_measured:
        exit_with_error(f'No valid PostgreSQL measurement files found for suffixes {measurement_suffixes}.')

    return all_measured


def _require_postgres(driver_type: DriverType):
    if driver_type != DriverType.POSTGRES:
        raise ValueError('Flat-feature tree models are currently implemented only for PostgreSQL datasets.')


def _warn_if_exists(path: str, label: str):
    if os.path.exists(path):
        print_warning(f'{label} already exists. It will be overwritten.')


if __name__ == '__main__':
    main()
