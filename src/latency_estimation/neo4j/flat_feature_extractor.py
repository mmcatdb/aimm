from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np


ROOT_NUMERIC_KEYS = ('EstimatedRows',)
NUMERIC_KEYS = ('EstimatedRows',)

CATEGORICAL_BASE_KEYS = (
    'operatorType',
    'operatorDepth',
    'operatorEdge',
    'planner',
    'runtime',
    'runtime-impl',
    'planner-impl',
)

SCHEMA_IDENTIFIER_CATEGORICAL_KEYS = (
    'LabelName',
    'Index',
)

RUNTIME_ONLY_KEYS = {
    'Rows',
    'DbHits',
    'PageCacheHits',
    'PageCacheMisses',
    'Time',
}


@dataclass
class DetailInfo:
    length: int = 0
    token_count: int = 0
    property_access_count: int = 0
    cache_property_count: int = 0
    label_count: int = 0
    relationship_count: int = 0
    variable_count: int = 0
    literal_count: int = 0
    arithmetic_count: int = 0
    has_predicate: bool = False
    has_comparison: bool = False
    has_index: bool = False
    has_unique: bool = False
    has_range: bool = False
    has_label_scan: bool = False
    has_relationship: bool = False
    has_property_access: bool = False
    has_cache_property: bool = False
    has_aggregation: bool = False
    has_order: bool = False
    has_limit: bool = False
    has_projection_alias: bool = False
    has_runtime_constant: bool = False
    has_subquery: bool = False


@dataclass
class FlatFeatureExtractor:
    """Fixed-length Neo4j plan feature extractor for EXPLAIN-only inference.

    Neo4j's PROFILE output contains runtime fields such as rows, db hits, page-cache
    counters, and operator time. This extractor deliberately ignores those fields so
    that training and prediction use only features available from plain EXPLAIN.
    """

    include_schema_identifiers: bool = False
    vocabularies: dict[str, set[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        for key in self.categorical_keys:
            self.vocabularies.setdefault(key, set())

    @property
    def categorical_keys(self) -> tuple[str, ...]:
        include_schema_identifiers = getattr(self, 'include_schema_identifiers', False)
        if include_schema_identifiers:
            return CATEGORICAL_BASE_KEYS + SCHEMA_IDENTIFIER_CATEGORICAL_KEYS
        return CATEGORICAL_BASE_KEYS

    def fit(self, plans: list[dict]) -> None:
        for plan in plans:
            for node, _ in self._iter_nodes(plan):
                for key, value in self._extract_categorical_values(node):
                    self.vocabularies.setdefault(key, set()).add(value)
            for key, value in self._extract_contextual_categorical_values(plan):
                self.vocabularies.setdefault(key, set()).add(value)

        self.feature_names = self._build_feature_names()

    def transform_plan(self, plan: dict) -> np.ndarray:
        if not self.feature_names:
            self.feature_names = self._build_feature_names()

        nodes = list(self._iter_nodes(plan))
        plan_nodes = [node for node, _ in nodes]
        root = plan_nodes[0] if plan_nodes else {}

        features = (
            self._numeric_root_features(root)
            + self._numeric_aggregate_features(plan_nodes)
            + self._estimated_row_flow_features(nodes)
            + self._structural_features(nodes)
            + self._operator_family_features(plan_nodes)
            + self._detail_features(plan_nodes)
            + self._identifier_features(plan_nodes)
            + self._categorical_count_features(plan_nodes)
        )

        if len(features) != len(self.feature_names):
            raise ValueError(
                f'Neo4j flat feature extractor produced {len(features)} features, but its schema has '
                f'{len(self.feature_names)} entries. Recreate the dataset and feature extractor.'
            )

        return np.array(features, dtype=np.float32)

    def transform_plans(self, plans: list[dict]) -> np.ndarray:
        return np.vstack([self.transform_plan(plan) for plan in plans])

    def _build_feature_names(self) -> list[str]:
        names = list[str]()

        for key in ROOT_NUMERIC_KEYS:
            names.append(f'root.{key}')
            names.append(f'root.log1p.{key}')

        for key in NUMERIC_KEYS:
            for agg in ('sum', 'max', 'mean', 'min', 'std'):
                names.append(f'plan.{agg}.{key}')
                names.append(f'plan.log1p.{agg}.{key}')

        names.extend([
            'plan.estimated_rows.root_to_max_ratio',
            'plan.estimated_rows.max_to_root_ratio',
            'plan.estimated_rows.sum_to_root_ratio',
            'plan.estimated_rows.leaf_sum',
            'plan.estimated_rows.log1p.leaf_sum',
            'plan.estimated_rows.expand_sum',
            'plan.estimated_rows.log1p.expand_sum',
            'plan.estimated_rows.scan_sum',
            'plan.estimated_rows.log1p.scan_sum',
            'plan.estimated_rows.join_sum',
            'plan.estimated_rows.log1p.join_sum',
            'plan.estimated_rows.max_parent_child_ratio',
            'plan.estimated_rows.mean_parent_child_ratio',
            'plan.node_count',
            'plan.log1p.node_count',
            'plan.max_depth',
            'plan.leaf_count',
            'plan.branch_count',
            'plan.binary_branch_count',
            'plan.max_children',
            'plan.avg_children',
            'plan.pipeline_depth',
            'plan.root_child_count',
            'plan.max_identifiers',
            'plan.query_is_ordered',
            'plan.root_ordered',
            'plan.scan_count',
            'plan.index_seek_count',
            'plan.unique_index_seek_count',
            'plan.label_scan_count',
            'plan.expand_count',
            'plan.expand_into_count',
            'plan.optional_expand_count',
            'plan.filter_count',
            'plan.join_count',
            'plan.apply_count',
            'plan.aggregation_count',
            'plan.sort_count',
            'plan.top_count',
            'plan.limit_count',
            'plan.projection_count',
            'plan.cache_properties_count',
            'plan.distinct_count',
            'plan.argument_count',
            'details.length_sum',
            'details.log1p.length_sum',
            'details.length_max',
            'details.token_count_sum',
            'details.property_access_count',
            'details.cache_property_count',
            'details.label_count',
            'details.relationship_count',
            'details.variable_count_sum',
            'details.literal_count_sum',
            'details.arithmetic_count_sum',
            'details.predicate_count',
            'details.comparison_count',
            'details.index_count',
            'details.unique_count',
            'details.range_count',
            'details.label_scan_count',
            'details.relationship_pattern_count',
            'details.property_access_node_count',
            'details.cache_property_node_count',
            'details.aggregation_count',
            'details.order_count',
            'details.limit_count',
            'details.projection_alias_count',
            'details.runtime_constant_count',
            'details.subquery_count',
            'identifiers.total_count',
            'identifiers.unique_count',
            'identifiers.max_per_node',
            'identifiers.mean_per_node',
            'identifiers.root_count',
        ])

        for key in self.categorical_keys:
            for value in sorted(self.vocabularies.get(key, set())):
                names.append(f'count.{key}.{value}')

        return names

    def _numeric_root_features(self, root: dict) -> list[float]:
        features = list[float]()
        for key in ROOT_NUMERIC_KEYS:
            value = self._numeric(self._args(root).get(key, 0.0))
            features.extend([value, math.log1p(max(0.0, value))])
        return features

    def _numeric_aggregate_features(self, nodes: list[dict]) -> list[float]:
        features = list[float]()
        for key in NUMERIC_KEYS:
            values = [self._numeric(self._args(node).get(key, 0.0)) for node in nodes]
            mean = sum(values) / len(values) if values else 0.0
            variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
            aggregates = [
                sum(values),
                max(values) if values else 0.0,
                mean,
                min(values) if values else 0.0,
                math.sqrt(variance),
            ]
            for value in aggregates:
                features.extend([value, math.log1p(max(0.0, value))])
        return features

    def _estimated_row_flow_features(self, nodes: list[tuple[dict, int]]) -> list[float]:
        if not nodes:
            return [0.0] * 13

        plan_nodes = [node for node, _ in nodes]
        root_rows = self._estimated_rows(plan_nodes[0])
        row_values = [self._estimated_rows(node) for node in plan_nodes]
        max_rows = max(row_values) if row_values else 0.0
        sum_rows = sum(row_values)
        leaf_sum = sum(self._estimated_rows(node) for node in plan_nodes if not self._children(node))
        expand_sum = sum(self._estimated_rows(node) for node in plan_nodes if self._is_expand(node))
        scan_sum = sum(self._estimated_rows(node) for node in plan_nodes if self._is_scan(node))
        join_sum = sum(self._estimated_rows(node) for node in plan_nodes if self._is_join(node))

        parent_child_ratios = []
        for node, _ in nodes:
            parent_rows = self._estimated_rows(node)
            for child in self._children(node):
                child_rows = self._estimated_rows(child)
                parent_child_ratios.append(self._ratio(parent_rows, child_rows))

        return [
            self._ratio(root_rows, max_rows),
            self._ratio(max_rows, root_rows),
            self._ratio(sum_rows, root_rows),
            leaf_sum,
            math.log1p(max(0.0, leaf_sum)),
            expand_sum,
            math.log1p(max(0.0, expand_sum)),
            scan_sum,
            math.log1p(max(0.0, scan_sum)),
            join_sum,
            math.log1p(max(0.0, join_sum)),
            max(parent_child_ratios) if parent_child_ratios else 0.0,
            sum(parent_child_ratios) / len(parent_child_ratios) if parent_child_ratios else 0.0,
        ]

    def _structural_features(self, nodes: list[tuple[dict, int]]) -> list[float]:
        node_count = len(nodes)
        depths = [depth for _, depth in nodes]
        plan_nodes = [node for node, _ in nodes]
        child_counts = [len(self._children(node)) for node in plan_nodes]
        identifier_counts = [len(node.get('identifiers') or []) for node in plan_nodes]
        root = plan_nodes[0] if plan_nodes else {}

        return [
            float(node_count),
            math.log1p(node_count),
            float(max(depths) if depths else 0),
            float(sum(1 for count in child_counts if count == 0)),
            float(sum(1 for count in child_counts if count > 0)),
            float(sum(1 for count in child_counts if count >= 2)),
            float(max(child_counts) if child_counts else 0),
            float(sum(child_counts) / len(child_counts) if child_counts else 0),
            float(self._longest_unary_chain(root)),
            float(len(self._children(root))),
            float(max(identifier_counts) if identifier_counts else 0),
            float(any(self._string(self._args(node).get('Order')) for node in plan_nodes)),
            float(bool(self._string(self._args(root).get('Order')))),
        ]

    def _operator_family_features(self, nodes: list[dict]) -> list[float]:
        return [
            float(sum(1 for node in nodes if self._is_scan(node))),
            float(sum(1 for node in nodes if self._is_index_seek(node))),
            float(sum(1 for node in nodes if 'UniqueIndexSeek' in self._op(node))),
            float(sum(1 for node in nodes if self._op(node) == 'NodeByLabelScan')),
            float(sum(1 for node in nodes if self._is_expand(node))),
            float(sum(1 for node in nodes if self._op(node) == 'Expand(Into)')),
            float(sum(1 for node in nodes if self._op(node).startswith('OptionalExpand'))),
            float(sum(1 for node in nodes if self._op(node) == 'Filter')),
            float(sum(1 for node in nodes if self._is_join(node))),
            float(sum(1 for node in nodes if 'Apply' in self._op(node))),
            float(sum(1 for node in nodes if 'Aggregation' in self._op(node))),
            float(sum(1 for node in nodes if 'Sort' in self._op(node))),
            float(sum(1 for node in nodes if self._op(node) in ('Top', 'PartialTop'))),
            float(sum(1 for node in nodes if self._op(node) == 'Limit')),
            float(sum(1 for node in nodes if self._op(node) == 'Projection')),
            float(sum(1 for node in nodes if self._op(node) == 'CacheProperties')),
            float(sum(1 for node in nodes if self._op(node) == 'Distinct')),
            float(sum(1 for node in nodes if self._op(node) == 'Argument')),
        ]

    def _detail_features(self, nodes: list[dict]) -> list[float]:
        infos = [self._parse_details(self._string(self._args(node).get('Details'))) for node in nodes]
        length_sum = sum(info.length for info in infos)

        return [
            float(length_sum),
            math.log1p(length_sum),
            float(max((info.length for info in infos), default=0)),
            float(sum(info.token_count for info in infos)),
            float(sum(info.property_access_count for info in infos)),
            float(sum(info.cache_property_count for info in infos)),
            float(sum(info.label_count for info in infos)),
            float(sum(info.relationship_count for info in infos)),
            float(sum(info.variable_count for info in infos)),
            float(sum(info.literal_count for info in infos)),
            float(sum(info.arithmetic_count for info in infos)),
            float(sum(info.has_predicate for info in infos)),
            float(sum(info.has_comparison for info in infos)),
            float(sum(info.has_index for info in infos)),
            float(sum(info.has_unique for info in infos)),
            float(sum(info.has_range for info in infos)),
            float(sum(info.has_label_scan for info in infos)),
            float(sum(info.has_relationship for info in infos)),
            float(sum(info.has_property_access for info in infos)),
            float(sum(info.has_cache_property for info in infos)),
            float(sum(info.has_aggregation for info in infos)),
            float(sum(info.has_order for info in infos)),
            float(sum(info.has_limit for info in infos)),
            float(sum(info.has_projection_alias for info in infos)),
            float(sum(info.has_runtime_constant for info in infos)),
            float(sum(info.has_subquery for info in infos)),
        ]

    def _identifier_features(self, nodes: list[dict]) -> list[float]:
        identifiers_per_node = [list(node.get('identifiers') or []) for node in nodes]
        all_identifiers = [identifier for identifiers in identifiers_per_node for identifier in identifiers]
        counts = [len(identifiers) for identifiers in identifiers_per_node]

        return [
            float(len(all_identifiers)),
            float(len(set(all_identifiers))),
            float(max(counts) if counts else 0),
            float(sum(counts) / len(counts) if counts else 0),
            float(counts[0] if counts else 0),
        ]

    def _categorical_count_features(self, nodes: list[dict]) -> list[float]:
        counters = {key: Counter[str]() for key in self.categorical_keys}
        for node in nodes:
            for key, value in self._extract_categorical_values(node):
                counters[key][value] += 1
        if nodes:
            for key, value in self._extract_contextual_categorical_values(nodes[0]):
                counters[key][value] += 1

        features = list[float]()
        for key in self.categorical_keys:
            for value in sorted(self.vocabularies.get(key, set())):
                features.append(float(counters[key][value]))
        return features

    def _extract_categorical_values(self, node: dict) -> list[tuple[str, str]]:
        output = list[tuple[str, str]]()
        args = self._args(node)

        op = self._op(node)
        if op:
            output.append(('operatorType', op))

        for key in self.categorical_keys:
            if key in ('operatorType', 'operatorDepth', 'operatorEdge'):
                continue
            value = self._string(args.get(key))
            if value:
                output.append((key, value))

        return output

    def _extract_contextual_categorical_values(self, root: dict) -> list[tuple[str, str]]:
        output = list[tuple[str, str]]()
        for node, depth in self._iter_nodes(root):
            op = self._op(node)
            depth_bucket = min(depth, 8)
            output.append(('operatorDepth', f'{depth_bucket}:{op}'))
            for child in self._children(node):
                output.append(('operatorEdge', f'{op}->{self._op(child)}'))
        return output

    def _parse_details(self, details: str) -> DetailInfo:
        info = DetailInfo(length=len(details))
        if not details:
            return info

        info.token_count = len(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', details))
        info.property_access_count = len(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b', details))
        info.cache_property_count = len(re.findall(r'cache\[[^\]]+\]', details))
        info.label_count = len(re.findall(r':[A-Za-z_][A-Za-z0-9_]*', details))
        info.relationship_count = len(re.findall(r'\[:[A-Za-z_][A-Za-z0-9_]*\]', details))
        info.variable_count = len(set(re.findall(r'\b[a-z][A-Za-z0-9_]*(?=[:.)\]])', details)))
        info.literal_count = len(re.findall(r'\$[A-Za-z_][A-Za-z0-9_]*|\b\d+(?:\.\d+)?\b', details))
        info.arithmetic_count = len(re.findall(r'(?<![A-Za-z])-|\+|\*|/', details))

        info.has_comparison = bool(re.search(r'<=|>=|<>|!=|=|<|>| CONTAINS | STARTS WITH | ENDS WITH ', details, re.IGNORECASE))
        info.has_predicate = info.has_comparison or ' WHERE ' in f' {details} '
        info.has_index = 'INDEX' in details.upper()
        info.has_unique = 'UNIQUE' in details.upper()
        info.has_range = 'RANGE' in details.upper()
        info.has_label_scan = 'SCAN' in details.upper()
        info.has_relationship = info.relationship_count > 0
        info.has_property_access = info.property_access_count > 0
        info.has_cache_property = info.cache_property_count > 0
        info.has_aggregation = bool(re.search(r'\b(count|sum|avg|min|max|collect)\s*\(', details, re.IGNORECASE))
        info.has_order = bool(re.search(r'\b(ASC|DESC)\b', details))
        info.has_limit = 'LIMIT' in details.upper()
        info.has_projection_alias = ' AS ' in f' {details} '
        info.has_runtime_constant = 'RuntimeConstant' in details
        info.has_subquery = 'Subquery' in details or 'CALL' in details.upper()
        return info

    def _iter_nodes(self, root: dict, depth: int = 0):
        yield root, depth
        for child in self._children(root):
            yield from self._iter_nodes(child, depth + 1)

    def _longest_unary_chain(self, node: dict) -> int:
        children = self._children(node)
        if len(children) != 1:
            return 1 if node else 0
        return 1 + self._longest_unary_chain(children[0])

    @staticmethod
    def _args(node: dict) -> dict:
        args = node.get('args')
        if args is None:
            args = node.get('arguments')
        return args if isinstance(args, dict) else {}

    @staticmethod
    def _children(node: dict) -> list[dict]:
        children = node.get('children', [])
        return children if isinstance(children, list) else []

    @staticmethod
    def _op(node: dict) -> str:
        return str(node.get('operatorType', 'Unknown')).replace('@neo4j', '')

    def _estimated_rows(self, node: dict) -> float:
        return self._numeric(self._args(node).get('EstimatedRows', 0.0))

    def _is_scan(self, node: dict) -> bool:
        op = self._op(node)
        return 'Scan' in op or 'IndexSeek' in op or op.startswith('NodeCount')

    def _is_index_seek(self, node: dict) -> bool:
        op = self._op(node)
        return 'IndexSeek' in op or 'IndexScan' in op

    def _is_expand(self, node: dict) -> bool:
        return 'Expand' in self._op(node)

    def _is_join(self, node: dict) -> bool:
        return 'Join' in self._op(node) or self._op(node) == 'TriadicSelection'

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _numeric(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _string(value: Any) -> str:
        if value is None:
            return ''
        if isinstance(value, list):
            return ', '.join(str(item) for item in value)
        return str(value)


def save_flat_feature_extractor(path: str, feature_extractor: FlatFeatureExtractor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(feature_extractor, file)


def load_flat_feature_extractor(path: str) -> FlatFeatureExtractor:
    with open(path, 'rb') as file:
        return pickle.load(file)
