from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from core.files import JsonEncoder, open_output
from core.utils import EPSILON
from latency_estimation.neo4j.flat_dataset import Neo4jFlatArrayDataset
from latency_estimation.neo4j.flat_feature_extractor import FlatFeatureExtractor


CALIBRATION_CHOICES = ('none', 'scale', 'log_isotonic', 'thresholded_log_isotonic')


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


@dataclass
class LatencyCalibrator:
    mode: str
    scale: float = 1.0
    isotonic: Any | None = None
    threshold: float | None = None
    validation_objective: float | None = None

    def predict(self, raw_predictions: np.ndarray) -> np.ndarray:
        raw = np.maximum(np.asarray(raw_predictions, dtype=np.float64), EPSILON)

        if self.mode == 'scale':
            return np.maximum(raw * self.scale, EPSILON)

        if self.mode == 'log_isotonic':
            return self._predict_isotonic(raw)

        if self.mode == 'thresholded_log_isotonic':
            calibrated = self._predict_isotonic(raw)
            if self.threshold is None:
                return calibrated
            return np.where(raw <= self.threshold, calibrated, raw)

        raise ValueError(f'Unsupported Neo4j latency calibration mode: {self.mode}')

    def describe(self) -> str:
        parts = [f'mode={self.mode}']
        if self.mode == 'scale':
            parts.append(f'scale={self.scale:.4f}')
        if self.threshold is not None:
            parts.append(f'threshold={self.threshold:.4f} ms')
        if self.validation_objective is not None:
            parts.append(f'validation_objective={self.validation_objective:.4f}')
        return ', '.join(parts)

    def _predict_isotonic(self, raw_predictions: np.ndarray) -> np.ndarray:
        if self.isotonic is None:
            raise ValueError(f'Neo4j calibration mode "{self.mode}" requires an isotonic model.')
        log_calibrated = self.isotonic.predict(np.log1p(raw_predictions))
        return np.maximum(np.expm1(log_calibrated), EPSILON)


class Neo4jFlatLatencyModel:
    def __init__(
        self,
        estimator,
        feature_extractor: FlatFeatureExtractor,
        model_id: str,
        model_type: str,
        calibrator: LatencyCalibrator | None = None,
    ):
        self.estimator = estimator
        self.feature_extractor = feature_extractor
        self.model_id = model_id
        self.model_type = model_type
        self.calibrator = calibrator

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
        target = np.log1p(y)
        if sample_weight is None:
            self.estimator.fit(x, target)
            return
        self.estimator.fit(x, target, sample_weight=sample_weight)

    def fit_calibrator(self, x: np.ndarray, y: np.ndarray, mode: str) -> LatencyCalibrator | None:
        self.calibrator = fit_latency_calibrator(self.predict_raw(x), y, mode)
        return self.calibrator

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        predictions = np.expm1(self.estimator.predict(x))
        return np.maximum(predictions, EPSILON)

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = self.predict_raw(x)
        calibrator = getattr(self, 'calibrator', None)
        if calibrator is None:
            return predictions
        return np.maximum(calibrator.predict(predictions), EPSILON)

    def predict_plan(self, plan: dict) -> float:
        features = self.feature_extractor.transform_plan(plan)
        return float(self.predict(features.reshape(1, -1))[0])


def fit_latency_calibrator(raw_predictions: np.ndarray, actual: np.ndarray, mode: str) -> LatencyCalibrator | None:
    mode = mode.replace('-', '_').lower()
    if mode == 'none':
        return None
    if mode not in CALIBRATION_CHOICES:
        raise ValueError(f'Unsupported Neo4j latency calibration mode: {mode}')

    raw = np.maximum(np.asarray(raw_predictions, dtype=np.float64), EPSILON)
    y = np.maximum(np.asarray(actual, dtype=np.float64), EPSILON)
    if raw.shape != y.shape:
        raise ValueError(f'Calibration predictions and labels must have the same shape, got {raw.shape} and {y.shape}.')

    if mode == 'scale':
        scale, objective = _select_scale(raw, y)
        return LatencyCalibrator(mode=mode, scale=scale, validation_objective=objective)

    isotonic = _fit_log_isotonic(raw, y)
    calibrated = _predict_log_isotonic(isotonic, raw)

    if mode == 'log_isotonic':
        return LatencyCalibrator(
            mode=mode,
            isotonic=isotonic,
            validation_objective=_calibration_objective(calibrated, y),
        )

    thresholds = np.unique(np.quantile(raw, np.linspace(0.1, 0.9, 9)))
    if len(thresholds) == 0:
        return LatencyCalibrator(
            mode=mode,
            isotonic=isotonic,
            threshold=None,
            validation_objective=_calibration_objective(calibrated, y),
        )

    best_threshold = float(thresholds[0])
    best_objective = float('inf')
    for threshold in thresholds:
        adjusted = np.where(raw <= threshold, calibrated, raw)
        objective = _calibration_objective(adjusted, y)
        if objective < best_objective:
            best_objective = objective
            best_threshold = float(threshold)

    return LatencyCalibrator(
        mode=mode,
        isotonic=isotonic,
        threshold=best_threshold,
        validation_objective=best_objective,
    )


def _fit_log_isotonic(raw_predictions: np.ndarray, actual: np.ndarray):
    if len(np.unique(raw_predictions)) < 2:
        from sklearn.isotonic import IsotonicRegression

        isotonic = IsotonicRegression(out_of_bounds='clip')
        constant_x = np.array([0.0, 1.0], dtype=np.float64)
        constant_y = np.array([np.median(np.log1p(actual)), np.median(np.log1p(actual))], dtype=np.float64)
        isotonic.fit(constant_x, constant_y)
        return isotonic

    from sklearn.isotonic import IsotonicRegression

    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(np.log1p(raw_predictions), np.log1p(actual))
    return isotonic


def _predict_log_isotonic(isotonic, raw_predictions: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(isotonic.predict(np.log1p(raw_predictions))), EPSILON)


def _select_scale(raw_predictions: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    candidate_scales = np.exp(np.linspace(np.log(0.05), np.log(5.0), 301))
    ratio_candidates = np.array([
        np.median(actual / raw_predictions),
        np.exp(np.median(np.log1p(actual) - np.log1p(raw_predictions))),
    ])
    candidate_scales = np.unique(np.concatenate([candidate_scales, ratio_candidates]))

    best_scale = 1.0
    best_objective = float('inf')
    for scale in candidate_scales:
        if not np.isfinite(scale) or scale <= 0:
            continue
        objective = _calibration_objective(raw_predictions * scale, actual)
        if objective < best_objective:
            best_objective = objective
            best_scale = float(scale)

    return best_scale, best_objective


def _calibration_objective(predicted: np.ndarray, actual: np.ndarray) -> float:
    q_values = _q_values(predicted, actual)
    return float(np.median(q_values) + 0.1 * np.mean(q_values) + 0.5 * (1.0 - np.mean(q_values <= 5.0)))


def _q_values(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    predicted = np.maximum(np.asarray(predicted, dtype=np.float64), EPSILON)
    actual = np.maximum(np.asarray(actual, dtype=np.float64), EPSILON)
    return np.maximum(predicted / actual, actual / predicted)


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


def transform_dataset_features(feature_extractor: FlatFeatureExtractor, dataset: Neo4jFlatArrayDataset, dataset_label: str) -> np.ndarray:
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
    print(f'  R <= 5.0: {metrics.r_within_5_0 * 100:.1f} %')


def save_flat_model(path: str, model: Neo4jFlatLatencyModel) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def load_flat_model(path: str) -> Neo4jFlatLatencyModel:
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_metrics(path: str, metrics: dict[str, FlatModelMetrics]) -> None:
    with open_output(path) as file:
        json.dump({key: asdict(value) for key, value in metrics.items()}, file, indent=4, cls=JsonEncoder)
