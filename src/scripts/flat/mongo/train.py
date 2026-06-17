import argparse
import os

import numpy as np

from core.config import Config
from core.drivers import DriverType
from core.utils import exit_with_exception, print_warning
from latency_estimation.dataset import create_dataset_id, parse_dataset_id
from latency_estimation.mongo.flat_dataset import load_mongo_flat_dataset
from latency_estimation.mongo.flat_feature_extractor import load_flat_feature_extractor
from latency_estimation.mongo.flat_model import (
    MongoFlatLatencyModel,
    compute_metrics,
    create_flat_estimator,
    normalize_model_type,
    print_metrics,
    save_flat_model,
    save_metrics,
    transform_dataset_features,
)
from providers.path_provider import PathProvider


def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Train a MongoDB flat-feature tree latency model.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    try:
        run(Config.load(), args)
    except Exception as e:
        exit_with_exception(e)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('model_id', nargs=1, help='Id of the model. Pattern: mongo/{model_name}.')
    parser.add_argument('train_dataset', type=str, help='Name of the flat training dataset.')
    parser.add_argument('val_dataset', type=str, help='Name of the flat validation dataset.')
    parser.add_argument('--model-type', default='random_forest', choices=['random_forest', 'random-forest', 'rf', 'extra_trees', 'extra-trees', 'et', 'decision_tree', 'decision-tree', 'dt', 'xgboost', 'xgb', 'tail_blend', 'tail-blend', 'blend'], help='Tree regressor family.')
    parser.add_argument('--n-estimators', type=int, default=500, help='Number of trees for ensemble models.')
    parser.add_argument('--max-depth', type=int, default=None, help='Maximum tree depth.')
    parser.add_argument('--min-samples-leaf', type=int, default=1, help='Minimum samples per leaf.')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='XGBoost learning rate.')
    parser.add_argument('--subsample', type=float, default=0.9, help='XGBoost row subsampling.')
    parser.add_argument('--colsample-bytree', type=float, default=0.9, help='XGBoost feature subsampling.')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs for supported estimators.')
    parser.add_argument('--seed', type=int, default=69, help='Random seed.')
    parser.add_argument('--sample-weight', default='none', choices=['none', 'log_latency', 'sqrt_latency', 'capped_latency'], help='Optional training-row weighting based only on measured training latency.')
    parser.add_argument('--dry-run', action='store_true', help='Only print dataset/model setup. Do not train.')


def run(config: Config, args: argparse.Namespace):
    pp = PathProvider(config)
    model_id = args.model_id[0]
    driver_type, _ = parse_dataset_id(model_id)
    if driver_type != DriverType.MONGO:
        raise ValueError('MongoDB flat-feature tree models require a model id with pattern mongo/{model_name}.')

    model_type = normalize_model_type(args.model_type)
    train_id = create_dataset_id(driver_type, args.train_dataset)
    val_id = create_dataset_id(driver_type, args.val_dataset)

    train_dataset = load_mongo_flat_dataset(pp.flat_dataset(train_id))
    val_dataset = load_mongo_flat_dataset(pp.flat_dataset(val_id))
    feature_extractor = load_flat_feature_extractor(pp.flat_feature_extractor(train_id))

    print(f'Model id: {model_id}')
    print(f'Model type: {model_type}')
    print(f'Training set: {len(train_dataset)} items')
    print(f'Validation set: {len(val_dataset)} items')
    print(f'Feature dimension: {len(feature_extractor.feature_names)}')

    output_path = pp.flat_model(model_id)
    if os.path.exists(output_path):
        print_warning(f'Flat model "{model_id}" already exists. It will be overwritten.')

    if args.dry_run:
        print('\nDry run completed. Exiting before training.')
        return

    estimator = create_flat_estimator(args, feature_extractor.feature_names)
    model = MongoFlatLatencyModel(estimator, feature_extractor, model_id, model_type)
    train_x = transform_dataset_features(model.feature_extractor, train_dataset, train_id)
    train_y = train_dataset.y()
    train_weight = create_sample_weight(train_y, args.sample_weight)
    val_x = transform_dataset_features(model.feature_extractor, val_dataset, val_id)
    val_y = val_dataset.y()

    print('\nTraining flat MongoDB model...')
    model.fit(train_x, train_y, sample_weight=train_weight)

    train_metrics = compute_metrics(model.predict(train_x), train_y)
    val_metrics = compute_metrics(model.predict(val_x), val_y)
    print_metrics('Training metrics', train_metrics)
    print_metrics('Validation metrics', val_metrics)

    save_flat_model(output_path, model)
    save_metrics(pp.flat_metrics(model_id), {
        'train': train_metrics,
        'validation': val_metrics,
    })
    print(f'\nSaved flat model to {output_path}')


def create_sample_weight(y, mode: str):
    if mode == 'none':
        return None
    if mode == 'log_latency':
        return np.log1p(y) + 1.0
    if mode == 'sqrt_latency':
        return np.sqrt(y + 1.0)
    if mode == 'capped_latency':
        return np.minimum(y, 200.0) + 1.0
    raise ValueError(f'Unsupported sample-weight mode: {mode}')


if __name__ == '__main__':
    main()
