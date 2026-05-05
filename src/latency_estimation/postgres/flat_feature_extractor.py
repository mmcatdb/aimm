from __future__ import annotations

import math
import os
import pickle
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


SAFE_NUMERIC_KEYS = (
    'Plan Width',
    'Plan Rows',
    'Total Cost',
    'Startup Cost',
)

ROOT_NUMERIC_KEYS = (
    'Startup Cost',
    'Total Cost',
    'Plan Rows',
    'Plan Width',
)

CATEGORICAL_BASE_KEYS = (
    'Node Type',
    'Scan Node Type',
    'Join Type',
    'Strategy',
    'Parent Relationship',
    'Partial Mode',
    'Operation',
)

SCHEMA_IDENTIFIER_CATEGORICAL_KEYS = (
    'Relation Name',
    'Index Name',
)

SCAN_NODE_TYPES = (
    'Seq Scan',
    'Index Scan',
    'Index Only Scan',
    'Bitmap Heap Scan',
    'Bitmap Index Scan',
    'Tid Scan',
    'Subquery Scan',
    'Function Scan',
    'Values Scan',
    'CTE Scan',
    'WorkTable Scan',
    'Foreign Scan',
    'Custom Scan',
    'TableFunc Scan',
)


def unwrap_plan(plan: dict) -> dict:
    """Return the root plan node for either wrapped EXPLAIN JSON or a bare Plan."""
    return plan.get('Plan', plan)


@dataclass
class FlatFeatureExtractor:
    """Fixed-length PostgreSQL plan feature extractor for EXPLAIN-only inference.

    The extractor deliberately ignores runtime-only EXPLAIN ANALYZE fields such as
    Actual Rows, Actual Total Time, buffer block counters, and execution timings.
    This keeps the training/inference feature contract honest: the same features
    can be produced from plain EXPLAIN without executing the query.
    """

    include_schema_identifiers: bool = False
    """Include table/index-name count features. Useful within one schema, usually harmful across schemas."""
    vocabularies: dict[str, set[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        for key in self.categorical_keys:
            self.vocabularies.setdefault(key, set())

    @property
    def categorical_keys(self) -> tuple[str, ...]:
        # Older pickles may not have include_schema_identifiers.
        include_schema_identifiers = getattr(self, 'include_schema_identifiers', False)
        if include_schema_identifiers:
            return CATEGORICAL_BASE_KEYS + SCHEMA_IDENTIFIER_CATEGORICAL_KEYS
        return CATEGORICAL_BASE_KEYS

    def fit(self, plans: list[dict]) -> None:
        for plan in plans:
            for node, _ in self._iter_nodes(unwrap_plan(plan)):
                for key, value in self._extract_categorical_values(node):
                    self.vocabularies.setdefault(key, set()).add(value)

        self.feature_names = self._build_feature_names()

    def transform_plan(self, plan: dict) -> np.ndarray:
        if not self.feature_names:
            self.feature_names = self._build_feature_names()

        root = unwrap_plan(plan)
        nodes = list(self._iter_nodes(root))
        root_features = self._numeric_root_features(root)
        numeric_features = self._numeric_aggregate_features([node for node, _ in nodes])
        structural_features = self._structural_features(nodes)
        categorical_features = self._categorical_count_features([node for node, _ in nodes])

        return np.array(
            root_features + numeric_features + structural_features + categorical_features,
            dtype=np.float32,
        )

    def transform_plans(self, plans: list[dict]) -> np.ndarray:
        return np.vstack([self.transform_plan(plan) for plan in plans])

    def _build_feature_names(self) -> list[str]:
        names = list[str]()

        for key in ROOT_NUMERIC_KEYS:
            names.append(f'root.{key}')
            names.append(f'root.log1p.{key}')

        for key in SAFE_NUMERIC_KEYS:
            for agg in ('sum', 'max', 'mean', 'min'):
                names.append(f'plan.{agg}.{key}')
                names.append(f'plan.log1p.{agg}.{key}')

        names.extend([
            'plan.node_count',
            'plan.log1p.node_count',
            'plan.max_depth',
            'plan.leaf_count',
            'plan.branch_count',
            'plan.scan_count',
            'plan.join_count',
            'plan.sort_count',
            'plan.aggregate_count',
            'plan.parallel_aware_count',
            'plan.async_capable_count',
            'plan.filter_count',
            'plan.index_cond_count',
            'plan.join_filter_count',
            'plan.hash_cond_count',
            'plan.recheck_cond_count',
            'plan.group_key_count',
            'plan.sort_key_count',
            'plan.workers_planned_sum',
        ])

        for key in self.categorical_keys:
            for value in sorted(self.vocabularies.get(key, set())):
                names.append(f'count.{key}.{value}')

        return names

    def _numeric_root_features(self, root: dict) -> list[float]:
        features = list[float]()
        for key in ROOT_NUMERIC_KEYS:
            value = self._numeric(root.get(key, 0.0))
            features.extend([value, math.log1p(max(0.0, value))])
        return features

    def _numeric_aggregate_features(self, nodes: list[dict]) -> list[float]:
        features = list[float]()
        for key in SAFE_NUMERIC_KEYS:
            values = [self._numeric(node.get(key, 0.0)) for node in nodes]
            aggregates = [
                sum(values),
                max(values) if values else 0.0,
                sum(values) / len(values) if values else 0.0,
                min(values) if values else 0.0,
            ]
            for value in aggregates:
                features.extend([value, math.log1p(max(0.0, value))])
        return features

    def _structural_features(self, nodes: list[tuple[dict, int]]) -> list[float]:
        node_count = len(nodes)
        depths = [depth for _, depth in nodes]
        plan_nodes = [node for node, _ in nodes]

        return [
            float(node_count),
            math.log1p(node_count),
            float(max(depths) if depths else 0),
            float(sum(1 for node in plan_nodes if not node.get('Plans'))),
            float(sum(1 for node in plan_nodes if node.get('Plans'))),
            float(sum(1 for node in plan_nodes if 'Scan' in str(node.get('Node Type', '')))),
            float(sum(1 for node in plan_nodes if 'Join' in str(node.get('Node Type', '')) or node.get('Node Type') == 'Nested Loop')),
            float(sum(1 for node in plan_nodes if 'Sort' in str(node.get('Node Type', '')))),
            float(sum(1 for node in plan_nodes if 'Aggregate' in str(node.get('Node Type', '')) or node.get('Node Type') == 'Group')),
            float(sum(1 for node in plan_nodes if bool(node.get('Parallel Aware', False)))),
            float(sum(1 for node in plan_nodes if bool(node.get('Async Capable', False)))),
            float(sum(1 for node in plan_nodes if 'Filter' in node)),
            float(sum(1 for node in plan_nodes if 'Index Cond' in node)),
            float(sum(1 for node in plan_nodes if 'Join Filter' in node)),
            float(sum(1 for node in plan_nodes if 'Hash Cond' in node)),
            float(sum(1 for node in plan_nodes if 'Recheck Cond' in node)),
            float(sum(self._list_length(node.get('Group Key')) for node in plan_nodes)),
            float(sum(self._list_length(node.get('Sort Key')) for node in plan_nodes)),
            float(sum(self._numeric(node.get('Workers Planned', 0.0)) for node in plan_nodes)),
        ]

    def _categorical_count_features(self, nodes: list[dict]) -> list[float]:
        counters = {key: Counter[str]() for key in self.categorical_keys}
        for node in nodes:
            for key, value in self._extract_categorical_values(node):
                counters[key][value] += 1

        features = list[float]()
        for key in self.categorical_keys:
            for value in sorted(self.vocabularies.get(key, set())):
                features.append(float(counters[key][value]))
        return features

    def _extract_categorical_values(self, node: dict) -> list[tuple[str, str]]:
        output = list[tuple[str, str]]()

        node_type = self._string(node.get('Node Type'))
        if node_type:
            output.append(('Node Type', node_type))
            if node_type in SCAN_NODE_TYPES or 'Scan' in node_type:
                output.append(('Scan Node Type', node_type))

        for key in self.categorical_keys:
            if key in ('Node Type', 'Scan Node Type'):
                continue
            value = self._string(node.get(key))
            if value:
                output.append((key, value))

        return output

    def _iter_nodes(self, root: dict, depth: int = 0):
        yield root, depth
        for child in root.get('Plans', []):
            yield from self._iter_nodes(child, depth + 1)

    @staticmethod
    def _numeric(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _string(value) -> str:
        if value is None:
            return ''
        if isinstance(value, list):
            return ', '.join(str(item) for item in value)
        return str(value)

    @staticmethod
    def _list_length(value) -> int:
        if value is None:
            return 0
        if isinstance(value, list):
            return len(value)
        return 1


def save_flat_feature_extractor(path: str, feature_extractor: FlatFeatureExtractor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(feature_extractor, file)


def load_flat_feature_extractor(path: str) -> FlatFeatureExtractor:
    with open(path, 'rb') as file:
        return pickle.load(file)
