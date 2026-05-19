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
        features.extend(self._query_shape_features(nodes, root_stats))
        features.extend(self._filter_features(nodes))
        features.extend(self._index_features(nodes, root_stats))
        features.extend(self._categorical_count_features(nodes, plan))

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

    def _query_shape_features(self, nodes: list[dict], root_stats: dict) -> list[float]:
        count = self._numeric(root_stats.get('count', 0.0))
        root_limit = self._first_positive([self._numeric(node.get('limitAmount', 0.0)) for node in nodes])
        has_collscan = any(self._node_stage(node) == 'COLLSCAN' for node in nodes)
        has_sort = any(self._is_sort_stage(self._node_stage(node)) for node in nodes)
        has_group = any(self._is_group_stage(self._node_stage(node)) for node in nodes)
        has_unwind = any(self._node_stage(node) == 'AGG_UNWIND' for node in nodes)
        estimated_input_docs = self._estimated_input_docs(nodes, root_stats)
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

    def _estimated_input_docs(self, nodes: list[dict], root_stats: dict) -> float:
        count = self._numeric(root_stats.get('count', 0.0))
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
