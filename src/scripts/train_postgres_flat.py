import argparse
import os

from core.config import Config
from core.drivers import DriverType
from core.utils import exit_with_exception, print_warning
from latency_estimation.dataset import create_dataset_id, parse_dataset_id
from latency_estimation.postgres.flat_dataset import load_flat_dataset
from latency_estimation.postgres.flat_feature_extractor import load_flat_feature_extractor
from latency_estimation.postgres.flat_model import (
    PostgresFlatLatencyModel,
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
    parser = argparse.ArgumentParser(description='Train a PostgreSQL flat-feature latency model.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    try:
        run(Config.load(), args)
    except Exception as e:
        exit_with_exception(e)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('model_id', nargs=1, help='Id of the model. Pattern: postgres/{model_name}.')
    parser.add_argument('train_dataset', type=str, help='Name of the flat training dataset.')
    parser.add_argument('val_dataset', type=str, help='Name of the flat validation dataset.')
    parser.add_argument('--model-type', default='random_forest', choices=['random_forest', 'random-forest', 'rf', 'xgboost', 'xgb'], help='Regressor family.')
    parser.add_argument('--n-estimators', type=int, default=300, help='Number of trees/boosting rounds.')
    parser.add_argument('--max-depth', type=int, default=None, help='Maximum tree depth.')
    parser.add_argument('--min-samples-leaf', type=int, default=1, help='Random forest minimum samples per leaf.')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='XGBoost learning rate.')
    parser.add_argument('--subsample', type=float, default=0.9, help='XGBoost row subsampling.')
    parser.add_argument('--colsample-bytree', type=float, default=0.9, help='XGBoost feature subsampling.')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Parallel jobs for supported estimators.')
    parser.add_argument('--seed', type=int, default=69, help='Random seed.')
    parser.add_argument('--dry-run', action='store_true', help='Only print dataset/model setup. Do not train.')


def run(config: Config, args: argparse.Namespace):
    pp = PathProvider(config)
    model_id = args.model_id[0]
    driver_type, _ = parse_dataset_id(model_id)
    if driver_type != DriverType.POSTGRES:
        raise ValueError('Flat-feature tree models are currently implemented only for PostgreSQL.')

    model_type = normalize_model_type(args.model_type)
    train_id = create_dataset_id(driver_type, args.train_dataset)
    val_id = create_dataset_id(driver_type, args.val_dataset)

    train_dataset = load_flat_dataset(pp.flat_dataset(train_id))
    val_dataset = load_flat_dataset(pp.flat_dataset(val_id))
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

    estimator = create_flat_estimator(args)
    model = PostgresFlatLatencyModel(estimator, feature_extractor, model_id, model_type)
    train_x = transform_dataset_features(model.feature_extractor, train_dataset, train_id)
    train_y = train_dataset.y()
    val_x = transform_dataset_features(model.feature_extractor, val_dataset, val_id)
    val_y = val_dataset.y()

    print('\nTraining flat model...')
    model.fit(train_x, train_y)

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


if __name__ == '__main__':
    main()
