from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from core.files import JsonEncoder, open_output
from core.utils import EPSILON
from latency_estimation.mongo.flat_dataset import MongoFlatArrayDataset
from latency_estimation.mongo.flat_feature_extractor import FlatFeatureExtractor


@dataclass
class FlatModelMetrics:
    mae: float
    median_ae: float
    mre: float
    median_q: float
    mean_q: float
    r_within_1_5: float
    r_within_2_0: float
    r_within_5_0: float


class MongoFlatLatencyModel:
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

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        if sample_weight is None:
            self.estimator.fit(x, np.log1p(y))
            return

        self.estimator.fit(x, np.log1p(y), sample_weight=sample_weight)

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.expm1(self.estimator.predict(x))
        return np.maximum(predictions, EPSILON)

    def predict_plan(self, plan: dict, global_stats: dict) -> float:
        features = self.feature_extractor.transform_plan(plan, global_stats)
        return float(self.predict(features.reshape(1, -1))[0])


class FeatureSubsetLatencyBlendRegressor:
    def __init__(
        self,
        components: Sequence[tuple[object, np.ndarray]],
        weights: Sequence[float] | None = None,
    ):
        self.components = list(components)
        if weights is None:
            weights = [1.0 / len(self.components)] * len(self.components)
        weight_array = np.array(weights, dtype=np.float64)
        self.weights = weight_array / np.sum(weight_array)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        for estimator, feature_indices in self.components:
            estimator.fit(x[:, feature_indices], y, sample_weight=sample_weight)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        latency_predictions = []
        for estimator, feature_indices in self.components:
            log_prediction = estimator.predict(x[:, feature_indices])
            latency_predictions.append(np.maximum(np.expm1(log_prediction), EPSILON))

        blended_latency = np.zeros(x.shape[0], dtype=np.float64)
        for weight, prediction in zip(self.weights, latency_predictions):
            blended_latency += weight * prediction
        return np.log1p(np.maximum(blended_latency, EPSILON))


def create_flat_estimator(args, feature_names: Sequence[str] | None = None):
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

    if model_type == 'extra_trees':
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )

    if model_type == 'decision_tree':
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
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

    if model_type == 'tail_blend':
        if feature_names is None:
            raise ValueError('tail_blend requires feature names for feature-subset selection.')

        from sklearn.ensemble import RandomForestRegressor
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError('XGBoost is not installed. Install the "xgboost" package or use another model type.') from e

        selected_work_features = {
            'agg.log1p.lookup_scan_work',
            'agg.log1p.max_pipeline_rows',
            'agg.pipeline_lookup_count',
            'agg.nonsargable_lookup_count',
            'agg.unwind_count',
            'agg.lookup_count',
        }
        rf_feature_indices = np.array([
            index
            for index, name in enumerate(feature_names)
            if not name.startswith('agg.') or name in selected_work_features
        ], dtype=np.int64)
        all_feature_indices = np.arange(len(feature_names), dtype=np.int64)

        return FeatureSubsetLatencyBlendRegressor([
            (
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=10,
                    random_state=args.seed,
                    n_jobs=args.n_jobs,
                ),
                rf_feature_indices,
            ),
            (
                XGBRegressor(
                    n_estimators=800,
                    max_depth=8,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective='reg:squarederror',
                    random_state=args.seed,
                    n_jobs=args.n_jobs,
                ),
                all_feature_indices,
            ),
        ])

    raise ValueError(f'Unsupported flat model type: {args.model_type}')


def normalize_model_type(model_type: str) -> str:
    normalized = model_type.replace('-', '_').lower()
    if normalized in ('rf', 'forest', 'random_forest'):
        return 'random_forest'
    if normalized in ('et', 'extra_trees', 'extratrees'):
        return 'extra_trees'
    if normalized in ('dt', 'tree', 'decision_tree'):
        return 'decision_tree'
    if normalized in ('xgb', 'xgboost'):
        return 'xgboost'
    if normalized in ('tail_blend', 'tailblend', 'blend'):
        return 'tail_blend'
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
        r_within_5_0=float(np.mean(q_values <= 5.0)),
    )


def transform_dataset_features(feature_extractor: FlatFeatureExtractor, dataset: MongoFlatArrayDataset, dataset_label: str) -> np.ndarray:
    samples = dataset.plan_samples()
    if samples is not None:
        return feature_extractor.transform_samples(samples)

    x = dataset.x()
    expected_dim = len(feature_extractor.feature_names)
    if x.shape[1] != expected_dim:
        raise ValueError(
            f'Flat dataset "{dataset_label}" has {x.shape[1]} cached features, but the model feature extractor expects '
            f'{expected_dim}. Recreate the flat dataset so it stores raw plans and global_stats, or recreate it with the '
            f'training feature extractor using --feature-extractor-dataset.'
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
    print(f'  R <= 5.0: {metrics.r_within_5_0 * 100:.1f} %')


def save_flat_model(path: str, model: MongoFlatLatencyModel) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_flat_model(path: str) -> MongoFlatLatencyModel:
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_metrics(path: str, metrics: dict[str, FlatModelMetrics]) -> None:
    with open_output(path) as file:
        json.dump({key: asdict(value) for key, value in metrics.items()}, file, indent=4, cls=JsonEncoder)
