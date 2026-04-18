from typing_extensions import override
import numpy as np
from collections import defaultdict
import math
from core.utils import EPSILON
from ..feature_extractor import PlanNode, BaseFeatureExtractor
from .plan_extractor import PlanExtractor

class FeatureExtractor(BaseFeatureExtractor):
    """
    Feature extraction from MongoDB query plans.
    Handles both Classic Engine (explainVersion '1') and SBE (explainVersion '2') plans.
    For SBE plans, we use the queryPlan JSON tree, not the slotBasedPlan text.

    Key differences from PostgreSQL:
    - No pre-execution cardinality estimates (no Plan Rows / Plan Width)
    - Filter predicates must be encoded to help the network learn selectivity
    - Collection stats serve as the baseline for data volume estimation
    """

    #region Setup

    def __init__(self):
        self.stage_vocab = set()
        self.collection_vocab = set()
        self.index_name_vocab = set()
        self.direction_map = {'forward': 1.0, 'backward': -1.0}
        self.numeric_stats = {}

    @override
    def extend_vocabularies(self, plans: list[dict], global_stats: dict):
        numeric_features = defaultdict(list)

        def traverse(node):
            stage = self.get_node_type(node)
            self.stage_vocab.add(stage)

            # Collect collection names
            # TODO not working right now because namespace is not present in the plan?
            ns = node.get('namespace', '')
            if ns:
                collection = ns.split('.')[-1] if '.' in ns else ns
                self.collection_vocab.add(collection)

            if 'indexName' in node:
                self.index_name_vocab.add(node['indexName'])

            # Numeric features for normalization
            for key in ['limitAmount', 'skipAmount', 'memLimit']:
                if key in node:
                    numeric_features[key].append(float(node[key]))

            # Recurse children
            for child in self.get_node_children(node):
                traverse(child)

        for plan in plans:
            traverse(plan)

        # Add collection names from stats if not yet seen
        for collection_name in global_stats:
            self.collection_vocab.add(collection_name)

        # FIXME There should be only addition, not mutation of existing values.

        # Compute normalization stats
        for key, values in numeric_features.items():
            self.numeric_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values) + EPSILON,
            }

        # Add collection-stats normalization
        for stat_key in ['count', 'size', 'avgObjSize', 'storageSize', 'nindexes']:
            vals = [s[stat_key] for s in global_stats.values() if stat_key in s]
            if vals:
                self.numeric_stats[f'coll_{stat_key}'] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals) + EPSILON,
                }

    #endregion

    @override
    def extract_plan(self, plan: dict) -> PlanNode:
        collection = plan[PlanExtractor.GLOBAL_STATS_COLLECTION_KEY]
        return self.__extract_plan_inner(plan, collection)

    def __extract_plan_inner(self, plan: dict, collection: str) -> PlanNode:
        node_type = self.get_node_type(plan)
        features = self.__extract_node_features(plan, collection)
        node = PlanNode(node_type, features, None)

        for child in self.get_node_children(plan):
            child_node = self.extract_plan(child)
            node.add_child(child_node)

        return node

    #region Features

    def __extract_node_features(self, node: dict, collection: str) -> np.ndarray:
        """Extract feature vector for a single plan node."""
        op_type = self.get_node_type(node)
        features = []

        # 1. Stage one-hot
        features.extend(self._one_hot(op_type, self.stage_vocab))

        # 2. Global collection features
        features.extend(self.__extract_global_features(collection))

        # 3. Stage-specific features
        if op_type == 'COLLSCAN':
            features.extend(self.__extract_collscan(node))
        elif op_type in ('IXSCAN', 'EXPRESS_IXSCAN'):
            features.extend(self.__extract_ixscan(node))
        elif op_type == 'FETCH':
            features.extend(self.__extract_fetch(node))
        elif op_type == 'SORT':
            features.extend(self.__extract_sort(node))
        elif op_type in ('LIMIT', 'SKIP'):
            features.extend(self.__extract_limit_skip(node, op_type))
        elif op_type in ('PROJECTION_SIMPLE', 'PROJECTION_DEFAULT', 'PROJECTION_COVERED'):
            features.extend(self.__extract_projection(node))
        elif op_type == 'GROUP':
            features.extend(self.__extract_group(node))
        elif op_type == 'EQ_LOOKUP':
            features.extend(self.__extract_eq_lookup(node))
        elif op_type in ('AND_HASH', 'AND_SORTED', 'OR', 'SORT_MERGE'):
            features.extend(self.__extract_merge_ops(node))
        else:
            # Generic fallback: filter + direction
            features.extend(self.__extract_generic(node))

        return np.array(features, dtype=np.float32)

    def __extract_collscan(self, node: dict) -> list[float]:
        feats = []
        feats.extend(self.__encode_filter(node.get('filter')))
        feats.append(self.direction_map.get(node.get('direction', 'forward'), 0.0))
        return feats

    def __extract_ixscan(self, node: dict) -> list[float]:
        feats = []
        # Index name one-hot
        feats.extend(self._one_hot(node.get('indexName', ''), self.index_name_vocab))
        # isMultiKey
        feats.append(1.0 if node.get('isMultiKey', False) else 0.0)
        # isUnique
        feats.append(1.0 if node.get('isUnique', False) else 0.0)
        # isSparse
        feats.append(1.0 if node.get('isSparse', False) else 0.0)
        # direction
        feats.append(self.direction_map.get(node.get('direction', 'forward'), 0.0))
        # Index bounds encoding
        feats.extend(self.__encode_index_bounds(node.get('indexBounds')))
        # Number of key fields in the index
        key_pattern = node.get('keyPattern', {})
        feats.append(self.__log_normalize(len(key_pattern)))
        return feats

    def __extract_fetch(self, node: dict) -> list[float]:
        feats = []
        feats.extend(self.__encode_filter(node.get('filter')))
        return feats

    def __extract_sort(self, node: dict) -> list[float]:
        feats = []
        sort_pattern = node.get('sortPattern', {})
        feats.append(self.__log_normalize(len(sort_pattern)))
        mem_limit = node.get('memLimit', 104857600)
        feats.append(self.__log_normalize(mem_limit))
        feats.append(self.__log_normalize(node.get('limitAmount', 0)))
        # sort type
        sort_type = node.get('type', 'simple')
        feats.append(1.0 if sort_type == 'simple' else 0.0)
        return feats

    def __extract_limit_skip(self, node: dict, stage: str) -> list[float]:
        feats = []
        amount = node.get('limitAmount', 0) if stage == 'LIMIT' else node.get('skipAmount', 0)
        feats.append(self.__log_normalize(amount))
        return feats

    def __extract_projection(self, node: dict) -> list[float]:
        feats = []
        transform = node.get('transformBy', {})
        feats.append(self.__log_normalize(len(transform)))
        return feats

    # Accumulator operators we track
    _GROUP_ACCUMULATORS = ['$sum', '$avg', '$min', '$max', '$first', '$last', '$push', '$addToSet', '$count', '$stdDevPop', '$stdDevSamp']

    def __extract_group(self, node: dict) -> list[float]:
        """
        Extract features for GROUP stage.
        Encodes: num accumulators, accumulator type presence vector,
        group key complexity, whether grouping by compound key.
        """
        feats = []
        # Try to parse group spec from node (SBE queryPlan)
        # In most explain output, GROUP has limited info in queryPlanner.
        # But we can encode whatever is available.

        # Number of output fields / accumulators
        num_accumulators = 0
        acc_vec = np.zeros(len(self._GROUP_ACCUMULATORS), dtype=np.float32)

        # In SBE plans, GROUP may have accumulators, slots, or other info
        # Try to get from the explain output's aggregate pipeline (if available via context)
        # For now, encode node-level info
        slots = node.get('slots', '')
        if isinstance(slots, str):
            num_accumulators = slots.count('$')

        # Count accumulator types if we can find accumulator specs
        accumulators = node.get('accumulators', [])
        if isinstance(accumulators, list):
            for acc in accumulators:
                if isinstance(acc, dict):
                    for op in acc.values():
                        if isinstance(op, str) and op in self._GROUP_ACCUMULATORS:
                            index = self._GROUP_ACCUMULATORS.index(op)
                            acc_vec[index] = 1.0
                            num_accumulators += 1

        feats.append(self.__log_normalize(num_accumulators))
        feats.extend(acc_vec.tolist())

        # Group key complexity (1 = simple, >1 = compound)
        group_by = node.get('groupBy', node.get('_id', ''))
        if isinstance(group_by, dict):
            feats.append(self.__log_normalize(len(group_by)))
        else:
            feats.append(0.0)

        return feats

    def __extract_eq_lookup(self, node: dict) -> list[float]:
        feats = []
        # Foreign collection one-hot
        foreign_ns = node.get('foreignCollection', '')
        foreign_coll = foreign_ns.split('.')[-1] if '.' in foreign_ns else foreign_ns
        feats.extend(self._one_hot(foreign_coll, self.collection_vocab))
        # Foreign collection stats
        feats.extend(self.__extract_global_features(foreign_coll))
        # Strategy
        strategy = node.get('strategy', '')
        strategies = ['NestedLoopJoin', 'IndexedLoopJoin', 'HashLookup']
        for s in strategies:
            feats.append(1.0 if strategy == s else 0.0)
        return feats

    def __extract_merge_ops(self, node: dict) -> list[float]:
        feats = []
        feats.extend(self.__encode_filter(node.get('filter')))
        return feats

    def __extract_generic(self, node: dict) -> list[float]:
        feats = []
        feats.extend(self.__encode_filter(node.get('filter')))
        feats.append(self.direction_map.get(node.get('direction', 'forward'), 0.0))
        return feats

    def __extract_global_features(self, collection: str) -> np.ndarray:
        """
        Global context features for the collection:
        count, size, avgObjSize, storageSize, nindexes (all log-normalized).
        """
        stats = self._global_stats.get(collection, {})
        feats = []
        for key in ['count', 'size', 'avgObjSize', 'storageSize', 'nindexes']:
            feats.append(self.__log_normalize(stats.get(key, 0)))
        return np.array(feats, dtype=np.float32)

    def __encode_filter(self, filter_doc: dict | None) -> np.ndarray:
        """
        Encode a filter predicate into a fixed-size feature vector.
        Returns: operator presence vector + tree depth + num predicates + has_regex
                 + numeric value summary (min, max, mean, count, range_width)
                 + num_fields + is_equality + num_in_values
        """
        op_vec = np.zeros(len(FILTER_OPS), dtype=np.float32)
        stats = {'depth': 0, 'num_predicates': 0, 'has_regex': 0}
        numeric_values = []
        num_fields = 0
        is_equality = 0.0
        num_in_values = 0

        def walk(doc: dict, depth: int):
            nonlocal num_fields, is_equality, num_in_values
            if not doc:
                return

            stats['depth'] = max(stats['depth'], depth)
            for k, v in doc.items():
                if k in FILTER_OPS:
                    index = FILTER_OPS.index(k)
                    op_vec[index] = 1.0
                    stats['num_predicates'] += 1
                    if k == '$regex':
                        stats['has_regex'] = 1
                    if k == '$eq':
                        is_equality = 1.0
                    # Extract numeric values from comparison operators
                    if k in ('$gt', '$gte', '$lt', '$lte', '$eq', '$ne'):
                        if isinstance(v, (int, float)):
                            numeric_values.append(float(v))
                        elif hasattr(v, 'timestamp'):
                            # datetime -> epoch seconds (normalized)
                            numeric_values.append(v.timestamp() / 1e9)
                    if k == '$in' and isinstance(v, list):
                        num_in_values = len(v)
                        for item in v:
                            if isinstance(item, (int, float)):
                                numeric_values.append(float(item))
                    if isinstance(v, dict):
                        walk(v, depth + 1)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                walk(item, depth + 1)
                elif not k.startswith('$'):
                    # This is a field name
                    num_fields += 1
                    # Check for implicit equality
                    if not isinstance(v, dict):
                        is_equality = 1.0
                        if isinstance(v, (int, float)):
                            numeric_values.append(float(v))
                        elif hasattr(v, 'timestamp'):
                            numeric_values.append(v.timestamp() / 1e9)
                    else:
                        walk(v, depth + 1)

        if filter_doc:
            walk(filter_doc, 0)

        # Numeric value summary features
        if numeric_values:
            nv = np.array(numeric_values)
            val_min = self.__log_normalize(abs(float(np.min(nv))))
            val_max = self.__log_normalize(abs(float(np.max(nv))))
            val_mean = self.__log_normalize(abs(float(np.mean(nv))))
            val_count = self.__log_normalize(len(nv))
            val_range = self.__log_normalize(abs(float(np.max(nv) - np.min(nv))))
        else:
            val_min = val_max = val_mean = val_count = val_range = 0.0

        extra = np.array([
            stats['depth'],
            self.__log_normalize(stats['num_predicates']),
            stats['has_regex'],
            val_min,
            val_max,
            val_mean,
            val_count,
            val_range,
            self.__log_normalize(num_fields),
            is_equality,
            self.__log_normalize(num_in_values),
        ], dtype=np.float32)
        return np.concatenate([op_vec, extra])

    def __encode_index_bounds(self, bounds: dict | None) -> np.ndarray:
        """
        Encode index bounds into a fixed-size vector:
        - num_bound_fields
        - total_intervals (sum of intervals across all fields)
        - is_point_lookup (all intervals are single points)
        - is_range (any interval is a range)
        """
        if not bounds:
            return np.zeros(4, dtype=np.float32)

        num_fields = len(bounds)
        total_intervals = 0
        is_point = 1.0
        is_range = 0.0
        for field, intervals in bounds.items():
            if isinstance(intervals, list):
                total_intervals += len(intervals)
                for iv in intervals:
                    if isinstance(iv, str):
                        # e.g. '[1, 1]' vs '[1, 100]'
                        parts = iv.strip('[]').split(',')
                        if len(parts) == 2:
                            lo, hi = parts[0].strip(), parts[1].strip()
                            if lo != hi:
                                is_point = 0.0
                                is_range = 1.0

        return np.array([
            self.__log_normalize(num_fields),
            self.__log_normalize(total_intervals),
            is_point,
            is_range,
        ], dtype=np.float32)

    def __log_normalize(self, value: float) -> float:
        """Log-scale normalization for wide-range values."""
        return math.log1p(max(0, value))

    #endregion

    @staticmethod
    @override
    def get_node_type(node: dict) -> str:
        return node.get('stage', 'UNKNOWN')

    @staticmethod
    @override
    def get_node_children(node: dict) -> list[dict]:
        # Handle both inputStage and inputStages
        children = []
        if 'inputStage' in node:
            children.append(node['inputStage'])
        if 'inputStages' in node:
            children.extend(node['inputStages'])
        return children

# All known MongoDB execution stages (not used right now)
# KNOWN_STAGES = [
#     'COLLSCAN', 'IXSCAN', 'FETCH', 'SORT', 'SORT_MERGE',
#     'LIMIT', 'SKIP', 'PROJECTION_SIMPLE', 'PROJECTION_DEFAULT',
#     'PROJECTION_COVERED', 'AND_HASH', 'AND_SORTED', 'OR',
#     'COUNT_SCAN', 'TEXT_MATCH', 'TEXT_OR', 'GEO_NEAR_2D',
#     'GEO_NEAR_2DSPHERE', 'SHARDING_FILTER', 'SUBPLAN',
#     'EQ_LOOKUP', 'GROUP', 'UNWIND', 'EOF',
#     # SBE-specific
#     'NLJ', 'HJ',  # nested loop join, hash join
# ]

# MongoDB filter operators
FILTER_OPS = [
    '$eq', '$ne', '$gt', '$gte', '$lt', '$lte',
    '$in', '$nin', '$regex', '$exists', '$type',
    '$all', '$elemMatch', '$size',
    '$and', '$or', '$nor', '$not',
]
