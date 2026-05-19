from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable

import numpy as np
from bson.decimal128 import Decimal128


GLOBAL_STATS_KEYS = (
    'count',
    'size',
    'avgObjSize',
    'storageSize',
    'nindexes',
    'totalIndexSize',
)

NODE_NUMERIC_KEYS = (
    'limitAmount',
    'skipAmount',
    'memLimit',
    'pipelineStageCount',
)

KNOWN_STAGE_GROUPS = (
    'COLLSCAN',
    'IXSCAN',
    'FETCH',
    'SORT',
    'LIMIT',
    'SKIP',
    'PROJECTION_SIMPLE',
    'PROJECTION_DEFAULT',
    'PROJECTION_COVERED',
    'GROUP',
    'EQ_LOOKUP',
    'OR',
    'AND_HASH',
    'AND_SORTED',
    'SORT_MERGE',
    'COUNT_SCAN',
    'IDHACK',
    'AGG_MATCH',
    'AGG_UNWIND',
    'AGG_PROJECT',
    'AGG_ADD_FIELDS',
    'AGG_SET',
    'AGG_UNSET',
    'AGG_SORT',
    'AGG_GROUP',
    'AGG_LOOKUP',
    'AGG_LIMIT',
    'AGG_SKIP',
    'AGG_COUNT',
    'AGG_FACET',
    'AGG_BUCKET',
)

FILTER_OPS = (
    '$eq',
    '$ne',
    '$gt',
    '$gte',
    '$lt',
    '$lte',
    '$in',
    '$nin',
    '$regex',
    '$exists',
    '$type',
    '$all',
    '$elemMatch',
    '$size',
    '$and',
    '$or',
    '$nor',
    '$not',
)

CATEGORICAL_BASE_KEYS = (
    'stage',
    'direction',
    'sort_type',
)

SCHEMA_IDENTIFIER_CATEGORICAL_KEYS = (
    'collection',
    'indexName',
    'indexField',
    'sortField',
    'filterField',
    'projectionField',
)

BOUND_NUMBER_RE = re.compile(r'-?\d+(?:\.\d+)?')
BOUND_QUOTED_STRING_RE = re.compile(r'"([^"]*)"')
UNBOUNDED_MARKERS = ('MinKey', 'MaxKey', '-inf', 'inf', 'Infinity', '9223372036854775807', '-9223372036854775808')
QUERY_FILTER_KEY = '$queryFilter'
QUERY_FILTERS_KEY = '$queryFilters'
FIELD_STATS_KEY = 'fieldStats'

RANGE_OPERATORS = ('$gt', '$gte', '$lt', '$lte')
EQUALITY_FALLBACK_SELECTIVITY = 0.10
RANGE_FALLBACK_SELECTIVITY = 0.33
REGEX_FALLBACK_SELECTIVITY = 0.10
EXISTS_FALLBACK_SELECTIVITY = 0.80


@dataclass
class IndexBoundsSummary:
    fields: int = 0
    intervals: int = 0
    point_intervals: int = 0
    range_intervals: int = 0
    unbounded_intervals: int = 0
    numeric_bounds: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)


@dataclass
class FilterSummary:
    op_counts: Counter[str] = field(default_factory=Counter)
    depth: int = 0
    predicates: int = 0
    fields: int = 0
    in_values: int = 0
    numeric_values: list[float] = field(default_factory=list)
    has_regex: bool = False


@dataclass
class SelectivitySummary:
    predicate_selectivities: list[float] = field(default_factory=list)
    fields: set[str] = field(default_factory=set)
    known_predicates: int = 0
    missing_predicates: int = 0
    equality_predicates: int = 0
    in_predicates: int = 0
    range_predicates: int = 0
    regex_predicates: int = 0
    exists_predicates: int = 0
    array_predicates: int = 0
    logical_or_count: int = 0
    logical_not_count: int = 0


@dataclass
class FlatFeatureExtractor:
    """Fixed-length MongoDB plan feature extractor for queryPlanner-only inference.

    Runtime-only fields from executionStats are intentionally ignored. Features are
    built from the safe plan shape/details returned by queryPlanner plus database
    global_stats that can be collected once per database before prediction.
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

    def fit(self, samples: list[tuple[dict, dict]]) -> None:
        for plan, global_stats in samples:
            for node, _ in self._iter_nodes(plan):
                for key, value in self._extract_categorical_values(node, plan):
                    self.vocabularies.setdefault(key, set()).add(value)

            if self.include_schema_identifiers:
                for collection in global_stats:
                    self.vocabularies.setdefault('collection', set()).add(collection)

        self.feature_names = self._build_feature_names()

    def transform_plan(self, plan: dict, global_stats: dict) -> np.ndarray:
        if not self.feature_names:
            self.feature_names = self._build_feature_names()

        nodes_with_depth = list(self._iter_nodes(plan))
        nodes = [node for node, _ in nodes_with_depth]
        root_collection = self._root_collection(plan)
        root_stats = self._stats_for_collection(global_stats, root_collection)

        features = []
        features.extend(self._global_features(global_stats, root_collection))
        features.extend(self._numeric_aggregate_features(nodes))
        features.extend(self._structural_features(nodes_with_depth))
        features.extend(self._stage_group_features(nodes))
        features.extend(self._query_shape_features(plan, nodes, root_stats))
        features.extend(self._filter_features(nodes))
        features.extend(self._selectivity_features(plan, nodes, root_stats))
        features.extend(self._index_features(nodes, root_stats))
        features.extend(self._categorical_count_features(nodes, plan))

        if len(features) != len(self.feature_names):
            rebuilt_names = self._build_feature_names()
            if len(features) != len(rebuilt_names):
                raise ValueError(
                    f'Mongo flat feature extractor produced {len(features)} features, but its feature-name schema has '
                    f'{len(self.feature_names)} entries. Recreate the Mongo flat dataset and feature extractor.'
                )
            self.feature_names = rebuilt_names

        return np.array(features, dtype=np.float32)

    def transform_samples(self, samples: list[tuple[dict, dict]]) -> np.ndarray:
        return np.vstack([self.transform_plan(plan, global_stats) for plan, global_stats in samples])

    def _build_feature_names(self) -> list[str]:
        names = list[str]()

        for prefix in ('root_collection', 'database_total', 'database_max'):
            for key in GLOBAL_STATS_KEYS:
                names.append(f'{prefix}.{key}')
                names.append(f'{prefix}.log1p.{key}')

        names.extend([
            'root_collection.count_fraction',
            'root_collection.size_fraction',
            'root_collection.index_size_fraction',
            'root_collection.indexes_per_million_docs',
            'root_collection.index_bytes_per_doc',
        ])

        for key in NODE_NUMERIC_KEYS:
            for agg in ('sum', 'max', 'mean', 'min'):
                names.append(f'plan.{agg}.{key}')
                names.append(f'plan.log1p.{agg}.{key}')

        names.extend([
            'plan.node_count',
            'plan.log1p.node_count',
            'plan.max_depth',
            'plan.leaf_count',
            'plan.branch_count',
            'plan.is_cached',
            'plan.has_index_scan',
            'plan.has_collection_scan',
            'plan.has_sort',
            'plan.has_group',
            'plan.has_limit',
            'plan.has_projection',
            'plan.root_is_collection_scan',
            'plan.root_is_index_scan',
            'plan.root_is_sort',
            'plan.root_is_group',
        ])

        for stage in KNOWN_STAGE_GROUPS:
            names.append(f'stage_count.{stage}')

        names.extend([
            'shape.root_limit',
            'shape.log1p.root_limit',
            'shape.limit_to_collection_count',
            'shape.min_limit_or_count',
            'shape.log1p.min_limit_or_count',
            'shape.collection_scan_work',
            'shape.log1p.collection_scan_work',
            'shape.sort_collection_work',
            'shape.log1p.sort_collection_work',
            'shape.group_collection_work',
            'shape.log1p.group_collection_work',
            'shape.projection_field_sum',
            'shape.log1p.projection_field_sum',
            'shape.sort_field_sum',
            'shape.log1p.sort_field_sum',
            'shape.unwind_input_work',
            'shape.log1p.unwind_input_work',
            'shape.group_key_field_sum',
            'shape.log1p.group_key_field_sum',
            'shape.accumulator_sum',
            'shape.log1p.accumulator_sum',
            'shape.expression_operator_count',
            'shape.log1p.expression_operator_count',
            'shape.array_expression_operator_count',
            'shape.log1p.array_expression_operator_count',
            'shape.date_expression_operator_count',
            'shape.log1p.date_expression_operator_count',
        ])

        for op in FILTER_OPS:
            names.append(f'filter.op_count.{op}')

        names.extend([
            'filter.max_depth',
            'filter.log1p.predicate_count',
            'filter.log1p.field_count',
            'filter.has_regex',
            'filter.log1p.in_value_count',
            'filter.log1p.numeric_count',
            'filter.numeric_min_abs_log1p',
            'filter.numeric_max_abs_log1p',
            'filter.numeric_mean_abs_log1p',
            'filter.numeric_range_abs_log1p',
        ])

        names.extend([
            'selectivity.has_field_stats',
            'selectivity.query_filter_count',
            'selectivity.predicate_count',
            'selectivity.known_predicate_count',
            'selectivity.missing_predicate_count',
            'selectivity.field_count',
            'selectivity.equality_predicate_count',
            'selectivity.in_predicate_count',
            'selectivity.range_predicate_count',
            'selectivity.regex_predicate_count',
            'selectivity.exists_predicate_count',
            'selectivity.array_predicate_count',
            'selectivity.logical_or_count',
            'selectivity.logical_not_count',
            'selectivity.min_predicate',
            'selectivity.max_predicate',
            'selectivity.mean_predicate',
            'selectivity.compound_filter',
            'selectivity.neg_log10_compound_filter',
            'selectivity.estimated_examined_docs',
            'selectivity.log1p.estimated_examined_docs',
            'selectivity.limit_adjusted_docs',
            'selectivity.log1p.limit_adjusted_docs',
            'selectivity.known_predicate_fraction',
            'selectivity.missing_predicate_fraction',
        ])

        names.extend([
            'index.scan_count',
            'index.unique_count',
            'index.multikey_count',
            'index.sparse_count',
            'index.partial_count',
            'index.key_field_sum',
            'index.log1p.key_field_sum',
            'index.bounds_field_sum',
            'index.log1p.bounds_field_sum',
            'index.interval_sum',
            'index.log1p.interval_sum',
            'index.point_interval_sum',
            'index.log1p.point_interval_sum',
            'index.range_interval_sum',
            'index.log1p.range_interval_sum',
            'index.unbounded_interval_sum',
            'index.bounds_numeric_min_abs_log1p',
            'index.bounds_numeric_max_abs_log1p',
            'index.bounds_numeric_mean_abs_log1p',
            'index.bounds_width_mean_log1p',
            'index.bounds_width_max_log1p',
            'index.heuristic_selectivity',
            'index.heuristic_examined_docs',
            'index.log1p.heuristic_examined_docs',
        ])

        for key in self.categorical_keys:
            for value in sorted(self.vocabularies.get(key, set())):
                names.append(f'count.{key}.{value}')

        return names

    def _global_features(self, global_stats: dict, root_collection: str) -> list[float]:
        root_stats = self._stats_for_collection(global_stats, root_collection)
        totals = {key: sum(self._numeric(stats.get(key, 0.0)) for stats in global_stats.values()) for key in GLOBAL_STATS_KEYS}
        max_values = {key: max([self._numeric(stats.get(key, 0.0)) for stats in global_stats.values()] or [0.0]) for key in GLOBAL_STATS_KEYS}

        features = []
        for stats in (root_stats, totals, max_values):
            for key in GLOBAL_STATS_KEYS:
                value = self._numeric(stats.get(key, 0.0))
                features.extend([value, math.log1p(max(0.0, value))])

        count = self._numeric(root_stats.get('count', 0.0))
        size = self._numeric(root_stats.get('size', 0.0))
        index_size = self._numeric(root_stats.get('totalIndexSize', 0.0))
        features.extend([
            self._ratio(count, totals['count']),
            self._ratio(size, totals['size']),
            self._ratio(index_size, totals['totalIndexSize']),
            self._ratio(self._numeric(root_stats.get('nindexes', 0.0)) * 1_000_000.0, count),
            self._ratio(index_size, count),
        ])
        return features

    def _numeric_aggregate_features(self, nodes: list[dict]) -> list[float]:
        features = []
        for key in NODE_NUMERIC_KEYS:
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

    def _structural_features(self, nodes_with_depth: list[tuple[dict, int]]) -> list[float]:
        node_count = len(nodes_with_depth)
        depths = [depth for _, depth in nodes_with_depth]
        nodes = [node for node, _ in nodes_with_depth]
        root_stage = self._node_stage(nodes[0]) if nodes else ''
        stage_counter = Counter(self._node_stage(node) for node in nodes)

        return [
            float(node_count),
            math.log1p(node_count),
            float(max(depths) if depths else 0),
            float(sum(1 for node in nodes if not self._node_children(node))),
            float(sum(1 for node in nodes if self._node_children(node))),
            float(bool(nodes and nodes[0].get('isCached', False))),
            float(stage_counter['IXSCAN'] > 0),
            float(stage_counter['COLLSCAN'] > 0),
            float(any(self._is_sort_stage(stage) for stage in stage_counter)),
            float(any(self._is_group_stage(stage) for stage in stage_counter)),
            float(any(self._is_limit_stage(stage) for stage in stage_counter)),
            float(any(self._is_projection_stage(stage) for stage in stage_counter)),
            float(root_stage == 'COLLSCAN'),
            float(root_stage == 'IXSCAN'),
            float(self._is_sort_stage(root_stage)),
            float(self._is_group_stage(root_stage)),
        ]

    def _stage_group_features(self, nodes: list[dict]) -> list[float]:
        stage_counter = Counter(self._node_stage(node) for node in nodes)
        return [float(stage_counter[stage]) for stage in KNOWN_STAGE_GROUPS]

    def _query_shape_features(self, root: dict, nodes: list[dict], root_stats: dict) -> list[float]:
        count = self._numeric(root_stats.get('count', 0.0))
        root_limit = self._first_positive([self._numeric(node.get('limitAmount', 0.0)) for node in nodes])
        has_collscan = any(self._node_stage(node) == 'COLLSCAN' for node in nodes)
        has_sort = any(self._is_sort_stage(self._node_stage(node)) for node in nodes)
        has_group = any(self._is_group_stage(self._node_stage(node)) for node in nodes)
        has_unwind = any(self._node_stage(node) == 'AGG_UNWIND' for node in nodes)
        estimated_input_docs = self._estimated_input_docs(root, nodes, root_stats)
        min_limit_or_count = min(root_limit, estimated_input_docs) if root_limit > 0 and estimated_input_docs > 0 else max(root_limit, estimated_input_docs)
        projection_fields = sum(len(node.get('transformBy', {})) for node in nodes if isinstance(node.get('transformBy'), dict))
        sort_fields = sum(len(node.get('sortPattern', {})) for node in nodes if isinstance(node.get('sortPattern'), dict))
        group_key_fields = sum(self._expression_field_count(node.get('groupBy')) for node in nodes if self._is_group_stage(self._node_stage(node)))
        accumulators = sum(len([key for key in node.get('accumulators', {}) if not key.startswith('$')]) for node in nodes if isinstance(node.get('accumulators'), dict))
        expression_summary = Counter[str]()
        for node in nodes:
            self._summarize_expression(node.get('transformBy'), expression_summary)
            self._summarize_expression(node.get('groupBy'), expression_summary)
            self._summarize_expression(node.get('accumulators'), expression_summary)
            self._summarize_expression(node.get('filter'), expression_summary)
        collscan_work = count if has_collscan else 0.0
        sort_work = estimated_input_docs if has_sort else 0.0
        group_work = estimated_input_docs if has_group else 0.0
        unwind_work = estimated_input_docs if has_unwind else 0.0
        expression_operator_count = sum(expression_summary.values())
        array_expression_operator_count = sum(expression_summary[op] for op in ('$size', '$filter', '$map', '$sortArray', '$slice'))
        date_expression_operator_count = sum(expression_summary[op] for op in ('$dateTrunc', '$dateDiff'))

        return [
            root_limit,
            math.log1p(root_limit),
            self._ratio(root_limit, count),
            min_limit_or_count,
            math.log1p(max(0.0, min_limit_or_count)),
            collscan_work,
            math.log1p(max(0.0, collscan_work)),
            sort_work,
            math.log1p(max(0.0, sort_work)),
            group_work,
            math.log1p(max(0.0, group_work)),
            float(projection_fields),
            math.log1p(projection_fields),
            float(sort_fields),
            math.log1p(sort_fields),
            unwind_work,
            math.log1p(max(0.0, unwind_work)),
            float(group_key_fields),
            math.log1p(group_key_fields),
            float(accumulators),
            math.log1p(accumulators),
            float(expression_operator_count),
            math.log1p(expression_operator_count),
            float(array_expression_operator_count),
            math.log1p(array_expression_operator_count),
            float(date_expression_operator_count),
            math.log1p(date_expression_operator_count),
        ]

    def _estimated_input_docs(self, root: dict, nodes: list[dict], root_stats: dict) -> float:
        count = self._numeric(root_stats.get('count', 0.0))
        query_filters = self._query_filter_documents(root, nodes)
        if query_filters:
            summary = SelectivitySummary()
            filter_selectivities = [
                self._estimate_filter_selectivity(filter_doc, root_stats, summary)
                for filter_doc in query_filters
            ]
            return count * self._combine_and_selectivities(filter_selectivities)

        ixscan_nodes = [node for node in nodes if self._node_stage(node) in ('IXSCAN', 'EXPRESS_IXSCAN')]
        if ixscan_nodes:
            summary = IndexBoundsSummary()
            for node in ixscan_nodes:
                node_summary = self._summarize_index_bounds(node.get('indexBounds'))
                summary.fields += node_summary.fields
                summary.intervals += node_summary.intervals
                summary.point_intervals += node_summary.point_intervals
                summary.range_intervals += node_summary.range_intervals
                summary.unbounded_intervals += node_summary.unbounded_intervals
                summary.numeric_bounds.extend(node_summary.numeric_bounds)
                summary.widths.extend(node_summary.widths)

            return count * self._heuristic_index_selectivity(summary, count)

        if any(self._node_stage(node) == 'COLLSCAN' for node in nodes):
            return count

        return count

    def _filter_features(self, nodes: list[dict]) -> list[float]:
        summary = FilterSummary()
        for node in nodes:
            self._summarize_filter(node.get('filter'), summary)

        features = [float(summary.op_counts[op]) for op in FILTER_OPS]
        numeric = summary.numeric_values
        if numeric:
            abs_values = [abs(value) for value in numeric]
            min_abs = math.log1p(min(abs_values))
            max_abs = math.log1p(max(abs_values))
            mean_abs = math.log1p(sum(abs_values) / len(abs_values))
            range_abs = math.log1p(max(numeric) - min(numeric)) if max(numeric) >= min(numeric) else 0.0
        else:
            min_abs = max_abs = mean_abs = range_abs = 0.0

        features.extend([
            float(summary.depth),
            math.log1p(summary.predicates),
            math.log1p(summary.fields),
            float(summary.has_regex),
            math.log1p(summary.in_values),
            math.log1p(len(numeric)),
            min_abs,
            max_abs,
            mean_abs,
            range_abs,
        ])
        return features

    def _selectivity_features(self, root: dict, nodes: list[dict], root_stats: dict) -> list[float]:
        count = self._numeric(root_stats.get('count', 0.0))
        query_filters = self._query_filter_documents(root, nodes)
        summary = SelectivitySummary()
        if query_filters:
            filter_selectivities = [
                self._estimate_filter_selectivity(filter_doc, root_stats, summary)
                for filter_doc in query_filters
            ]
            compound_selectivity = self._combine_and_selectivities(filter_selectivities)
        else:
            compound_selectivity = 1.0

        predicate_count = len(summary.predicate_selectivities)
        if predicate_count:
            min_predicate = min(summary.predicate_selectivities)
            max_predicate = max(summary.predicate_selectivities)
            mean_predicate = sum(summary.predicate_selectivities) / predicate_count
        else:
            min_predicate = max_predicate = mean_predicate = 1.0

        estimated_examined = count * compound_selectivity
        root_limit = self._first_positive([self._numeric(node.get('limitAmount', 0.0)) for node in nodes])
        limit_adjusted_docs = min(root_limit, estimated_examined) if root_limit > 0.0 else estimated_examined
        known_fraction = self._ratio(summary.known_predicates, predicate_count)
        missing_fraction = self._ratio(summary.missing_predicates, predicate_count)

        return [
            float(bool(self._field_stats_map(root_stats))),
            float(len(query_filters)),
            float(predicate_count),
            float(summary.known_predicates),
            float(summary.missing_predicates),
            float(len(summary.fields)),
            float(summary.equality_predicates),
            float(summary.in_predicates),
            float(summary.range_predicates),
            float(summary.regex_predicates),
            float(summary.exists_predicates),
            float(summary.array_predicates),
            float(summary.logical_or_count),
            float(summary.logical_not_count),
            min_predicate,
            max_predicate,
            mean_predicate,
            compound_selectivity,
            -math.log10(max(compound_selectivity, 1.0e-12)),
            estimated_examined,
            math.log1p(max(0.0, estimated_examined)),
            limit_adjusted_docs,
            math.log1p(max(0.0, limit_adjusted_docs)),
            known_fraction,
            missing_fraction,
        ]

    def _query_filter_documents(self, root: dict, nodes: list[dict]) -> list[dict]:
        query_filters = root.get(QUERY_FILTERS_KEY)
        if isinstance(query_filters, list):
            filters = [filter_doc for filter_doc in query_filters if isinstance(filter_doc, dict) and filter_doc]
            if filters:
                return filters

        query_filter = root.get(QUERY_FILTER_KEY)
        if isinstance(query_filter, dict) and query_filter:
            return [query_filter]

        # Backward-compatible fallback for old cached measurements that do not
        # have the original query filters attached to the root plan.
        return [
            node.get('filter')
            for node in nodes
            if isinstance(node.get('filter'), dict) and node.get('filter')
        ]

    def _estimate_filter_selectivity(self, filter_doc, root_stats: dict, summary: SelectivitySummary, field_prefix: str = '') -> float:
        if not isinstance(filter_doc, dict) or not filter_doc:
            return 1.0

        parts = []
        for key, child in filter_doc.items():
            if key == '$and' and isinstance(child, list):
                parts.append(self._combine_and_selectivities([
                    self._estimate_filter_selectivity(item, root_stats, summary, field_prefix)
                    for item in child
                ]))
            elif key == '$or' and isinstance(child, list):
                summary.logical_or_count += 1
                parts.append(self._combine_or_selectivities([
                    self._estimate_filter_selectivity(item, root_stats, summary, field_prefix)
                    for item in child
                ]))
            elif key == '$nor' and isinstance(child, list):
                summary.logical_not_count += 1
                or_selectivity = self._combine_or_selectivities([
                    self._estimate_filter_selectivity(item, root_stats, summary, field_prefix)
                    for item in child
                ])
                parts.append(self._clamp_selectivity(1.0 - or_selectivity))
            elif key == '$not':
                summary.logical_not_count += 1
                parts.append(self._clamp_selectivity(
                    1.0 - self._estimate_filter_selectivity(child, root_stats, summary, field_prefix)
                ))
            elif key.startswith('$'):
                parts.append(1.0)
            else:
                field = f'{field_prefix}.{key}' if field_prefix else key
                parts.append(self._estimate_field_selectivity(field, child, root_stats, summary))

        return self._combine_and_selectivities(parts)

    def _estimate_field_selectivity(self, field: str, expression, root_stats: dict, summary: SelectivitySummary) -> float:
        summary.fields.add(field)

        if not isinstance(expression, dict):
            return self._record_predicate(
                summary,
                '$eq',
                *self._estimate_equality_selectivity(field, expression, root_stats),
            )

        operator_keys = [key for key in expression if key.startswith('$')]
        if not operator_keys:
            return self._record_predicate(
                summary,
                '$eq',
                *self._estimate_equality_selectivity(field, expression, root_stats),
            )

        parts = []
        lower_bound = None
        lower_inclusive = True
        upper_bound = None
        upper_inclusive = True

        for operator, value in expression.items():
            if operator == '$eq':
                parts.append(self._record_predicate(
                    summary,
                    operator,
                    *self._estimate_equality_selectivity(field, value, root_stats),
                ))
            elif operator == '$ne':
                selectivity, known = self._estimate_equality_selectivity(field, value, root_stats)
                parts.append(self._record_predicate(summary, operator, self._clamp_selectivity(1.0 - selectivity), known))
            elif operator == '$in':
                parts.append(self._record_predicate(
                    summary,
                    operator,
                    *self._estimate_in_selectivity(field, value, root_stats),
                ))
            elif operator == '$nin':
                selectivity, known = self._estimate_in_selectivity(field, value, root_stats)
                parts.append(self._record_predicate(summary, operator, self._clamp_selectivity(1.0 - selectivity), known))
            elif operator in RANGE_OPERATORS:
                if operator in ('$gt', '$gte'):
                    lower_bound = value
                    lower_inclusive = operator == '$gte'
                else:
                    upper_bound = value
                    upper_inclusive = operator == '$lte'
            elif operator == '$exists':
                parts.append(self._record_predicate(
                    summary,
                    operator,
                    *self._estimate_exists_selectivity(field, bool(value), root_stats),
                ))
            elif operator == '$regex':
                parts.append(self._record_predicate(
                    summary,
                    operator,
                    *self._estimate_regex_selectivity(field, value, root_stats),
                ))
            elif operator == '$elemMatch':
                array_selectivity, array_known = self._estimate_exists_selectivity(field, True, root_stats)
                child_selectivity = self._estimate_filter_selectivity(value, root_stats, summary, field)
                known = array_known or self._field_stats(root_stats, field) is not None
                parts.append(self._record_predicate(
                    summary,
                    operator,
                    self._clamp_selectivity(min(array_selectivity, child_selectivity)),
                    known,
                ))
            elif operator == '$all':
                parts.append(self._record_predicate(
                    summary,
                    operator,
                    *self._estimate_all_selectivity(field, value, root_stats),
                ))
            elif operator == '$size':
                parts.append(self._record_predicate(
                    summary,
                    operator,
                    *self._estimate_array_size_selectivity(field, value, root_stats),
                ))
            elif operator == '$not':
                summary.logical_not_count += 1
                nested = self._estimate_field_selectivity(field, value, root_stats, summary)
                parts.append(self._clamp_selectivity(1.0 - nested))
            else:
                parts.append(self._record_predicate(summary, operator, EQUALITY_FALLBACK_SELECTIVITY, False))

        if lower_bound is not None or upper_bound is not None:
            parts.append(self._record_predicate(
                summary,
                '$range',
                *self._estimate_range_selectivity(
                    field,
                    lower_bound,
                    lower_inclusive,
                    upper_bound,
                    upper_inclusive,
                    root_stats,
                ),
            ))

        return self._combine_and_selectivities(parts)

    def _record_predicate(self, summary: SelectivitySummary, operator: str, selectivity: float, known: bool) -> float:
        selectivity = self._clamp_selectivity(selectivity)
        summary.predicate_selectivities.append(selectivity)
        if known:
            summary.known_predicates += 1
        else:
            summary.missing_predicates += 1

        if operator in ('$eq', '$ne'):
            summary.equality_predicates += 1
        elif operator in ('$in', '$nin', '$all'):
            summary.in_predicates += 1
        elif operator in RANGE_OPERATORS or operator == '$range':
            summary.range_predicates += 1
        elif operator == '$regex':
            summary.regex_predicates += 1
        elif operator == '$exists':
            summary.exists_predicates += 1
        elif operator in ('$elemMatch', '$size'):
            summary.array_predicates += 1

        return selectivity

    def _estimate_equality_selectivity(self, field: str, value, root_stats: dict) -> tuple[float, bool]:
        stats = self._field_stats(root_stats, field)
        count = max(1.0, self._numeric(root_stats.get('count', 0.0)))
        if not stats:
            return EQUALITY_FALLBACK_SELECTIVITY, False

        if bool(stats.get('is_unique', False)):
            return min(1.0, 1.0 / count), True

        top_count = self._top_value_count(stats, value)
        non_null_count = self._numeric(stats.get('non_null_count', stats.get('value_count', 0.0)))
        if top_count is not None:
            return min(1.0, self._ratio(min(top_count, non_null_count or top_count), count)), True

        distinct_count = self._numeric(stats.get('distinct_count', 0.0))
        if distinct_count > 0.0:
            top_total = sum(self._numeric(item.get('count', 0.0)) for item in stats.get('top_values', []) if isinstance(item, dict))
            remaining_count = max(0.0, non_null_count - top_total)
            remaining_distinct = max(1.0, distinct_count - len(stats.get('top_values', [])))
            estimated_count = remaining_count / remaining_distinct if remaining_count > 0.0 else non_null_count / distinct_count
            return min(1.0, max(1.0 / count, estimated_count / count)), True

        if non_null_count > 0.0:
            return min(1.0, max(1.0 / count, 1.0 / math.sqrt(non_null_count))), True

        return EQUALITY_FALLBACK_SELECTIVITY, False

    def _estimate_in_selectivity(self, field: str, values, root_stats: dict) -> tuple[float, bool]:
        if not isinstance(values, list):
            return self._estimate_equality_selectivity(field, values, root_stats)

        unique_values = []
        seen = set[str]()
        for value in values:
            key = repr(value)
            if key in seen:
                continue
            seen.add(key)
            unique_values.append(value)

        if not unique_values:
            return 0.0, True

        estimates = [self._estimate_equality_selectivity(field, value, root_stats) for value in unique_values]
        selectivity = sum(value for value, _ in estimates)
        exists_selectivity, exists_known = self._estimate_exists_selectivity(field, True, root_stats)
        if exists_known:
            selectivity = min(selectivity, exists_selectivity)
        return self._clamp_selectivity(selectivity), all(known for _, known in estimates) or exists_known

    def _estimate_all_selectivity(self, field: str, values, root_stats: dict) -> tuple[float, bool]:
        if not isinstance(values, list) or not values:
            return EQUALITY_FALLBACK_SELECTIVITY, False
        estimates = [self._estimate_equality_selectivity(field, value, root_stats) for value in values]
        return self._combine_and_selectivities([value for value, _ in estimates]), all(known for _, known in estimates)

    def _estimate_exists_selectivity(self, field: str, exists: bool, root_stats: dict) -> tuple[float, bool]:
        stats = self._field_stats(root_stats, field)
        count = max(1.0, self._numeric(root_stats.get('count', 0.0)))
        if not stats:
            selectivity = EXISTS_FALLBACK_SELECTIVITY if exists else 1.0 - EXISTS_FALLBACK_SELECTIVITY
            return self._clamp_selectivity(selectivity), False

        non_null = self._numeric(stats.get('non_null_count', 0.0))
        selectivity = self._ratio(non_null, count)
        if not exists:
            selectivity = 1.0 - selectivity
        return self._clamp_selectivity(selectivity), True

    def _estimate_range_selectivity(
        self,
        field: str,
        lower_bound,
        lower_inclusive: bool,
        upper_bound,
        upper_inclusive: bool,
        root_stats: dict,
    ) -> tuple[float, bool]:
        stats = self._field_stats(root_stats, field)
        count = max(1.0, self._numeric(root_stats.get('count', 0.0)))
        if not stats:
            return RANGE_FALLBACK_SELECTIVITY, False

        low = self._comparable_value(lower_bound)
        high = self._comparable_value(upper_bound)
        if low is not None and high is not None and low > high:
            return 0.0, True

        histogram = stats.get('histogram')
        if isinstance(histogram, list) and histogram:
            selected = 0.0
            total = 0.0
            for bucket in histogram:
                if not isinstance(bucket, dict):
                    continue
                bucket_low = self._comparable_value(bucket.get('min'))
                bucket_high = self._comparable_value(bucket.get('max'))
                bucket_count = self._numeric(bucket.get('count', 0.0))
                if bucket_low is None or bucket_high is None or bucket_count <= 0.0:
                    continue
                total += bucket_count
                selected += bucket_count * self._range_overlap_fraction(bucket_low, bucket_high, low, high)
            if total > 0.0:
                exists_selectivity = self._ratio(self._numeric(stats.get('non_null_count', total)), count)
                return min(exists_selectivity, selected / count), True

        min_value = self._comparable_value(stats.get('min'))
        max_value = self._comparable_value(stats.get('max'))
        if min_value is None or max_value is None:
            return RANGE_FALLBACK_SELECTIVITY, False

        fraction = self._range_overlap_fraction(min_value, max_value, low, high)
        exists_selectivity = self._ratio(self._numeric(stats.get('non_null_count', count)), count)
        return self._clamp_selectivity(fraction * exists_selectivity), True

    def _estimate_regex_selectivity(self, field: str, pattern, root_stats: dict) -> tuple[float, bool]:
        stats = self._field_stats(root_stats, field)
        if not stats:
            return REGEX_FALLBACK_SELECTIVITY, False

        prefix = self._regex_prefix(pattern)
        if not prefix:
            return REGEX_FALLBACK_SELECTIVITY, True

        count = max(1.0, self._numeric(root_stats.get('count', 0.0)))
        top_match_count = 0.0
        top_total = 0.0
        for item in stats.get('top_values', []):
            if not isinstance(item, dict):
                continue
            value = item.get('value')
            item_count = self._numeric(item.get('count', 0.0))
            top_total += item_count
            if isinstance(value, str) and value.startswith(prefix):
                top_match_count += item_count

        remaining = max(0.0, self._numeric(stats.get('non_null_count', 0.0)) - top_total)
        return self._clamp_selectivity((top_match_count + remaining * 0.05) / count), True

    def _estimate_array_size_selectivity(self, field: str, value, root_stats: dict) -> tuple[float, bool]:
        stats = self._field_stats(root_stats, field)
        size = self._numeric(value)
        if not stats:
            return EQUALITY_FALLBACK_SELECTIVITY, False

        count = max(1.0, self._numeric(root_stats.get('count', 0.0)))
        array_count = self._numeric(stats.get('array_count', 0.0))
        min_length = self._numeric(stats.get('array_min_length', 0.0))
        max_length = self._numeric(stats.get('array_max_length', 0.0))
        if array_count <= 0.0:
            return 0.0, True
        if size < min_length or (max_length > 0.0 and size > max_length):
            return 0.0, True
        length_span = max(1.0, max_length - min_length + 1.0)
        return self._clamp_selectivity((array_count / length_span) / count), True

    def _range_overlap_fraction(self, min_value: float, max_value: float, low: float | None, high: float | None) -> float:
        if max_value < min_value:
            return 0.0
        if max_value == min_value:
            if low is not None and min_value < low:
                return 0.0
            if high is not None and min_value > high:
                return 0.0
            return 1.0

        selected_low = min_value if low is None else max(min_value, low)
        selected_high = max_value if high is None else min(max_value, high)
        if selected_high < selected_low:
            return 0.0
        return self._clamp_selectivity((selected_high - selected_low) / (max_value - min_value))

    def _combine_and_selectivities(self, selectivities: list[float]) -> float:
        if not selectivities:
            return 1.0
        result = 1.0
        for index, selectivity in enumerate(sorted(self._clamp_selectivity(value) for value in selectivities)):
            exponent = 1.0 if index == 0 else 0.75
            result *= selectivity ** exponent
        return self._clamp_selectivity(result)

    def _combine_or_selectivities(self, selectivities: list[float]) -> float:
        if not selectivities:
            return 1.0
        complement = 1.0
        for selectivity in selectivities:
            complement *= 1.0 - self._clamp_selectivity(selectivity)
        return self._clamp_selectivity(1.0 - complement)

    def _field_stats(self, root_stats: dict, field: str) -> dict | None:
        stats = self._field_stats_map(root_stats).get(field)
        return stats if isinstance(stats, dict) else None

    def _field_stats_map(self, root_stats: dict) -> dict:
        stats = root_stats.get(FIELD_STATS_KEY, {}) if isinstance(root_stats, dict) else {}
        return stats if isinstance(stats, dict) else {}

    def _top_value_count(self, stats: dict, value) -> float | None:
        for item in stats.get('top_values', []):
            if not isinstance(item, dict):
                continue
            if self._values_equal(item.get('value'), value):
                return self._numeric(item.get('count', 0.0))
        return None

    def _values_equal(self, left, right) -> bool:
        if left == right:
            return True
        left_value = self._comparable_value(left)
        right_value = self._comparable_value(right)
        if left_value is None or right_value is None:
            return False
        return abs(left_value - right_value) <= 1.0e-9

    def _comparable_value(self, value) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, Decimal128):
            return float(value.to_decimal())
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, date):
            return datetime(value.year, value.month, value.day).timestamp()
        return None

    def _regex_prefix(self, pattern) -> str:
        if not isinstance(pattern, str) or not pattern.startswith('^'):
            return ''
        output = []
        escaped = False
        for char in pattern[1:]:
            if escaped:
                output.append(char)
                escaped = False
                continue
            if char == '\\':
                escaped = True
                continue
            if char in '.+*?[](){}|$':
                break
            output.append(char)
        return ''.join(output)

    @staticmethod
    def _clamp_selectivity(value: float) -> float:
        if math.isnan(value) or math.isinf(value):
            return 1.0
        return min(1.0, max(0.0, float(value)))

    def _index_features(self, nodes: list[dict], root_stats: dict) -> list[float]:
        ixscan_nodes = [node for node in nodes if self._node_stage(node) in ('IXSCAN', 'EXPRESS_IXSCAN')]
        summary = IndexBoundsSummary()

        for node in ixscan_nodes:
            node_summary = self._summarize_index_bounds(node.get('indexBounds'))
            summary.fields += node_summary.fields
            summary.intervals += node_summary.intervals
            summary.point_intervals += node_summary.point_intervals
            summary.range_intervals += node_summary.range_intervals
            summary.unbounded_intervals += node_summary.unbounded_intervals
            summary.numeric_bounds.extend(node_summary.numeric_bounds)
            summary.widths.extend(node_summary.widths)

        key_field_sum = sum(len(node.get('keyPattern', {})) for node in ixscan_nodes if isinstance(node.get('keyPattern'), dict))
        count = self._numeric(root_stats.get('count', 0.0))
        heuristic_selectivity = self._heuristic_index_selectivity(summary, count) if ixscan_nodes else 0.0
        heuristic_examined = count * heuristic_selectivity if ixscan_nodes else 0.0

        bounds_abs = [abs(value) for value in summary.numeric_bounds]
        width_values = summary.widths
        return [
            float(len(ixscan_nodes)),
            float(sum(1 for node in ixscan_nodes if bool(node.get('isUnique', False)))),
            float(sum(1 for node in ixscan_nodes if bool(node.get('isMultiKey', False)))),
            float(sum(1 for node in ixscan_nodes if bool(node.get('isSparse', False)))),
            float(sum(1 for node in ixscan_nodes if bool(node.get('isPartial', False)))),
            float(key_field_sum),
            math.log1p(key_field_sum),
            float(summary.fields),
            math.log1p(summary.fields),
            float(summary.intervals),
            math.log1p(summary.intervals),
            float(summary.point_intervals),
            math.log1p(summary.point_intervals),
            float(summary.range_intervals),
            math.log1p(summary.range_intervals),
            float(summary.unbounded_intervals),
            math.log1p(min(bounds_abs)) if bounds_abs else 0.0,
            math.log1p(max(bounds_abs)) if bounds_abs else 0.0,
            math.log1p(sum(bounds_abs) / len(bounds_abs)) if bounds_abs else 0.0,
            math.log1p(sum(width_values) / len(width_values)) if width_values else 0.0,
            math.log1p(max(width_values)) if width_values else 0.0,
            heuristic_selectivity,
            heuristic_examined,
            math.log1p(max(0.0, heuristic_examined)),
        ]

    def _categorical_count_features(self, nodes: list[dict], root: dict) -> list[float]:
        counters = {key: Counter[str]() for key in self.categorical_keys}
        for node in nodes:
            for key, value in self._extract_categorical_values(node, root):
                counters[key][value] += 1

        features = []
        for key in self.categorical_keys:
            for value in sorted(self.vocabularies.get(key, set())):
                features.append(float(counters[key][value]))
        return features

    def _extract_categorical_values(self, node: dict, root: dict) -> list[tuple[str, str]]:
        output = []
        stage = self._node_stage(node)
        if stage:
            output.append(('stage', stage))

        direction = self._string(node.get('direction'))
        if direction:
            output.append(('direction', direction))

        sort_type = self._string(node.get('type')) if stage == 'SORT' else ''
        if sort_type:
            output.append(('sort_type', sort_type))

        if not self.include_schema_identifiers:
            return output

        collection = self._root_collection(root)
        if collection:
            output.append(('collection', collection))

        index_name = self._string(node.get('indexName'))
        if index_name:
            output.append(('indexName', index_name))

        for field in self._mapping_keys(node.get('keyPattern')):
            output.append(('indexField', field))
        for field in self._mapping_keys(node.get('sortPattern')):
            output.append(('sortField', field))
        for field in self._mapping_keys(node.get('transformBy')):
            output.append(('projectionField', field))
        for field in self._filter_fields(node.get('filter')):
            output.append(('filterField', field))

        return output

    def _summarize_filter(self, value, summary: FilterSummary, depth: int = 0) -> None:
        if value is None:
            return
        summary.depth = max(summary.depth, depth)

        if isinstance(value, dict):
            for key, child in value.items():
                if key in FILTER_OPS:
                    summary.op_counts[key] += 1
                    summary.predicates += 1
                    if key == '$regex':
                        summary.has_regex = True
                    if key in ('$in', '$nin', '$all') and isinstance(child, list):
                        summary.in_values += len(child)
                    self._collect_numeric_values(child, summary.numeric_values)
                    self._summarize_filter(child, summary, depth + 1)
                elif key.startswith('$'):
                    self._summarize_filter(child, summary, depth + 1)
                else:
                    summary.fields += 1
                    if not isinstance(child, dict):
                        summary.op_counts['$eq'] += 1
                        summary.predicates += 1
                    self._collect_numeric_values(child, summary.numeric_values)
                    self._summarize_filter(child, summary, depth + 1)
        elif isinstance(value, list):
            for child in value:
                self._summarize_filter(child, summary, depth + 1)

    def _summarize_index_bounds(self, bounds) -> IndexBoundsSummary:
        summary = IndexBoundsSummary()
        if not isinstance(bounds, dict):
            return summary

        summary.fields = len(bounds)
        for intervals in bounds.values():
            if not isinstance(intervals, list):
                continue
            summary.intervals += len(intervals)
            for interval in intervals:
                parsed = self._parse_bound_interval(interval)
                summary.numeric_bounds.extend(parsed[0])
                if parsed[1] is not None:
                    summary.widths.append(parsed[1])
                if parsed[2]:
                    summary.unbounded_intervals += 1
                if parsed[3]:
                    summary.point_intervals += 1
                else:
                    summary.range_intervals += 1

        return summary

    def _parse_bound_interval(self, interval) -> tuple[list[float], float | None, bool, bool]:
        text = self._string(interval)
        if not text:
            return [], None, False, False

        unbounded = any(marker in text for marker in UNBOUNDED_MARKERS)
        quoted_values = BOUND_QUOTED_STRING_RE.findall(text)
        if len(quoted_values) >= 2 and quoted_values[0] == quoted_values[-1] and not unbounded:
            return [], 0.0, False, True

        values = []
        for match in BOUND_NUMBER_RE.findall(text):
            try:
                value = float(match)
            except ValueError:
                continue
            if abs(value) >= 9.0e18:
                unbounded = True
                continue
            values.append(value)

        if len(values) >= 2:
            low, high = values[0], values[-1]
            width = abs(high - low)
            point = width == 0.0 and not unbounded
            return values, width, unbounded, point

        return values, None, unbounded, False

    def _heuristic_index_selectivity(self, summary: IndexBoundsSummary, collection_count: float) -> float:
        if summary.intervals <= 0:
            return 1.0
        if summary.point_intervals == summary.intervals:
            if collection_count <= 0:
                return 0.01
            return min(1.0, max(1.0, summary.point_intervals) / collection_count)

        range_fraction = 0.5 if summary.unbounded_intervals else 0.25
        point_multiplier = 0.1 ** min(summary.point_intervals, 6)
        point_fraction = self._ratio(summary.point_intervals, collection_count)
        interval_penalty = min(1.0, math.log1p(summary.intervals) / 10.0)
        if summary.point_intervals > 0:
            interval_penalty *= min(1.0, max(point_multiplier, point_fraction))
        return min(1.0, max(0.000001, range_fraction * point_multiplier + point_fraction + interval_penalty * 0.05))

    def _iter_nodes(self, root: dict, depth: int = 0):
        yield root, depth
        for child in self._node_children(root):
            yield from self._iter_nodes(child, depth + 1)

    def _node_children(self, node: dict) -> list[dict]:
        children = []
        child = node.get('inputStage')
        if isinstance(child, dict):
            children.append(child)
        input_stages = node.get('inputStages')
        if isinstance(input_stages, list):
            children.extend(child for child in input_stages if isinstance(child, dict))
        return children

    def _node_stage(self, node: dict) -> str:
        return self._string(node.get('stage')) or 'UNKNOWN'

    def _is_sort_stage(self, stage: str) -> bool:
        return stage in ('SORT', 'SORT_MERGE', 'AGG_SORT')

    def _is_group_stage(self, stage: str) -> bool:
        return stage in ('GROUP', 'AGG_GROUP', 'AGG_BUCKET', 'AGG_BUCKET_AUTO')

    def _is_limit_stage(self, stage: str) -> bool:
        return stage in ('LIMIT', 'AGG_LIMIT')

    def _is_projection_stage(self, stage: str) -> bool:
        return stage.startswith('PROJECTION') or stage in ('AGG_PROJECT', 'AGG_ADD_FIELDS', 'AGG_SET', 'AGG_UNSET')

    def _root_collection(self, plan: dict) -> str:
        return self._string(plan.get('$collection'))

    def _stats_for_collection(self, global_stats: dict, collection: str) -> dict:
        stats = global_stats.get(collection, {}) if isinstance(global_stats, dict) else {}
        return stats if isinstance(stats, dict) else {}

    def _filter_fields(self, value) -> Iterable[str]:
        if isinstance(value, dict):
            for key, child in value.items():
                if not key.startswith('$'):
                    yield key
                yield from self._filter_fields(child)
        elif isinstance(value, list):
            for child in value:
                yield from self._filter_fields(child)

    def _collect_numeric_values(self, value, output: list[float]) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            output.append(float(value))
            return
        if isinstance(value, (datetime, date)):
            # Absolute calendar values do not transfer well across schemas. The
            # plan shape and index bounds carry the useful selectivity signal.
            return
        if isinstance(value, dict):
            for child in value.values():
                self._collect_numeric_values(child, output)
            return
        if isinstance(value, list):
            for child in value:
                self._collect_numeric_values(child, output)

    def _summarize_expression(self, value, output: Counter[str]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.startswith('$'):
                    output[key] += 1
                self._summarize_expression(child, output)
            return

        if isinstance(value, list):
            for child in value:
                self._summarize_expression(child, output)

    def _expression_field_count(self, value) -> int:
        if isinstance(value, dict):
            non_operator_keys = [key for key in value if not key.startswith('$')]
            return len(non_operator_keys) if non_operator_keys else 1
        if value is None:
            return 0
        return 1

    @staticmethod
    def _mapping_keys(value) -> list[str]:
        return list(value.keys()) if isinstance(value, dict) else []

    @staticmethod
    def _first_positive(values: list[float]) -> float:
        positives = [value for value in values if value > 0.0]
        return min(positives) if positives else 0.0

    @staticmethod
    def _numeric(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    @staticmethod
    def _string(value) -> str:
        if value is None:
            return ''
        return str(value)


def save_flat_feature_extractor(path: str, feature_extractor: FlatFeatureExtractor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(feature_extractor, file)


def load_flat_feature_extractor(path: str) -> FlatFeatureExtractor:
    with open(path, 'rb') as file:
        return pickle.load(file)
