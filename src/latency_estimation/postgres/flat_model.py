from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass

import numpy as np
from core.utils import EPSILON, JsonEncoder
from latency_estimation.postgres.flat_dataset import FlatArrayDataset
from latency_estimation.postgres.flat_feature_extractor import FlatFeatureExtractor


@dataclass
class FlatModelMetrics:
    mae: float
    median_ae: float
    mre: float
    median_q: float
    mean_q: float
    r_within_1_5: float
    r_within_2_0: float


class PostgresFlatLatencyModel:
    def __init__(
        self,
        estimator,
        feature_extractor: FlatFeatureExtractor,
        model_id: str,
        model_type: str,
    ):
        self.estimator = estimator
        self.feature_extractor = feature_extractor
        self.model_id = model_id
        self.model_type = model_type

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.estimator.fit(x, np.log1p(y))

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.expm1(self.estimator.predict(x))
        return np.maximum(predictions, EPSILON)

    def predict_plan(self, plan: dict) -> float:
        features = self.feature_extractor.transform_plan(plan)
        return float(self.predict(features.reshape(1, -1))[0])


def create_flat_estimator(args):
    model_type = normalize_model_type(args.model_type)

    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )

    if model_type == 'xgboost':
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError('XGBoost is not installed. Install the "xgboost" package or use --model-type random_forest.') from e

        return XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth is not None else 6,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            objective='reg:squarederror',
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )

    raise ValueError(f'Unsupported flat model type: {args.model_type}')


def normalize_model_type(model_type: str) -> str:
    normalized = model_type.replace('-', '_').lower()
    if normalized in ('rf', 'forest', 'random_forest'):
        return 'random_forest'
    if normalized in ('xgb', 'xgboost'):
        return 'xgboost'
    raise ValueError(f'Unsupported flat model type: {model_type}')


def compute_metrics(predicted: np.ndarray, actual: np.ndarray) -> FlatModelMetrics:
    absolute = np.abs(predicted - actual)
    relative = absolute / (actual + EPSILON)
    q_values = np.maximum(
        predicted / (actual + EPSILON),
        actual / (predicted + EPSILON),
    )

    return FlatModelMetrics(
        mae=float(np.mean(absolute)),
        median_ae=float(np.median(absolute)),
        mre=float(np.mean(relative)),
        median_q=float(np.median(q_values)),
        mean_q=float(np.mean(q_values)),
        r_within_1_5=float(np.mean(q_values <= 1.5)),
        r_within_2_0=float(np.mean(q_values <= 2.0)),
    )


def transform_dataset_features(feature_extractor: FlatFeatureExtractor, dataset: FlatArrayDataset, dataset_label: str) -> np.ndarray:
    plans = dataset.plans()
    if plans is not None:
        return feature_extractor.transform_plans(plans)

    x = dataset.x()
    expected_dim = len(feature_extractor.feature_names)
    if x.shape[1] != expected_dim:
        raise ValueError(
            f'Flat dataset "{dataset_label}" has {x.shape[1]} cached features, but the model feature extractor expects '
            f'{expected_dim}. Recreate the flat dataset so it stores raw plans, or recreate it with the training '
            f'feature extractor using --feature-extractor-dataset.'
        )

    return x


def print_metrics(title: str, metrics: FlatModelMetrics) -> None:
    print(f'\n{title}:')
    print(f'  Mean Absolute Error: {metrics.mae:.2f} ms')
    print(f'  Median Absolute Error: {metrics.median_ae:.2f} ms')
    print(f'  Mean Relative Error: {metrics.mre:.4f}')
    print(f'  Median R-value: {metrics.median_q:.2f}')
    print(f'  Mean R-value: {metrics.mean_q:.2f}')
    print(f'  R <= 1.5: {metrics.r_within_1_5 * 100:.1f} %')
    print(f'  R <= 2.0: {metrics.r_within_2_0 * 100:.1f} %')


def save_flat_model(path: str, model: PostgresFlatLatencyModel) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_flat_model(path: str) -> PostgresFlatLatencyModel:
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_metrics(path: str, metrics: dict[str, FlatModelMetrics]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        json.dump({key: asdict(value) for key, value in metrics.items()}, file, indent=4, cls=JsonEncoder)
