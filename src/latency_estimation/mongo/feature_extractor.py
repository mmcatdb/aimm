from typing_extensions import override
import numpy as np
from collections import defaultdict
import math
from common.utils import EPSILON
from common.drivers import MongoDriver
from common.database import MongoQuery, MongoAggregateQuery
from latency_estimation.feature_extractor import BaseFeatureExtractor

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

    def __init__(self):
        self.stage_vocab = set()
        self.collection_vocab = set()
        self.index_name_vocab = set()
        self.direction_map = {'forward': 1.0, 'backward': -1.0}
        self.numeric_stats = {}
        # Collection stats cache (loaded before training/inference)
        self.collection_stats = dict[str, dict[str, int | float]]()

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

    # TODO Not sure about this - why does mongo has these stats but postgres / neo4j don't?
    def build_collection_stats(self, driver: MongoDriver):
        """Get statistics for all collections."""
        db = driver.database()
        stats = {}
        for name in db.list_collection_names():
            if not name.startswith("system."):
                stats[name] = self.__get_collection_stats(driver, name)

        self.collection_stats = stats

    @staticmethod
    def __get_collection_stats(driver: MongoDriver, collection_name: str) -> dict[str, int | float]:
        """Get collection statistics via collStats command."""
        db = driver.database()
        stats = db.command("collStats", collection_name)
        return {
            'count': stats.get('count', 0),
            'size': stats.get('size', 0),
            'avgObjSize': stats.get('avgObjSize', 0),
            'storageSize': stats.get('storageSize', 0),
            'nindexes': stats.get('nindexes', 0),
            'totalIndexSize': stats.get('totalIndexSize', 0),
        }

    def build_vocabularies(self, plans: list[dict]):
        """
        Build vocabularies by scanning all plan trees.
        plans: list of plan root nodes (the winningPlan / queryPlan dicts).
        """
        numeric_features = defaultdict(list)

        def traverse(node):
            stage = self.get_node_type(node)
            self.stage_vocab.add(stage)

            # Collect collection names
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
        for collection_name in self.collection_stats:
            self.collection_vocab.add(collection_name)

        # Compute normalization stats
        for key, values in numeric_features.items():
            self.numeric_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values) + EPSILON,
            }

        # Add collection-stats normalization
        for stat_key in ['count', 'size', 'avgObjSize', 'storageSize', 'nindexes']:
            vals = [s[stat_key] for s in self.collection_stats.values() if stat_key in s]
            if vals:
                self.numeric_stats[f'coll_{stat_key}'] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals) + EPSILON,
                }

        print(f'Built vocabularies:')
        print(f'  Stages: {sorted(self.stage_vocab)}')
        print(f'  Collections: {sorted(self.collection_vocab)}')
        print(f'  Indexes: {sorted(self.index_name_vocab)}')

    def extract_features(self, node: dict, collection_name: str) -> np.ndarray:
        """Extract feature vector for a single plan node."""
        op_type = self.get_node_type(node)
        features = []

        # 1. Stage one-hot
        features.extend(self.__one_hot(op_type, self.stage_vocab))

        # 2. Global collection features
        features.extend(self.__extract_global_features(collection_name))

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

    @override
    def get_feature_dim(self, op_type: str) -> int:
        base_dim = len(self.stage_vocab) + self.__dim_global_features()

        if op_type == 'COLLSCAN':
            return base_dim + self.__dim_collscan()
        elif op_type in ('IXSCAN', 'EXPRESS_IXSCAN'):
            return base_dim + self.__dim_ixscan()
        elif op_type == 'FETCH':
            return base_dim + self.__dim_fetch()
        elif op_type == 'SORT':
            return base_dim + self.__dim_sort()
        elif op_type in ('LIMIT', 'SKIP'):
            return base_dim + self.__dim_limit_skip()
        elif op_type in ('PROJECTION_SIMPLE', 'PROJECTION_DEFAULT', 'PROJECTION_COVERED'):
            return base_dim + self.__dim_projection()
        elif op_type == 'GROUP':
            return base_dim + self.__dim_group()
        elif op_type == 'EQ_LOOKUP':
            return base_dim + self.__dim_eq_lookup()
        elif op_type in ('AND_HASH', 'AND_SORTED', 'OR', 'SORT_MERGE'):
            return base_dim + self.__dim_merge_ops()
        else:
            return base_dim + self.__dim_generic()

    #region COLLSCAN

    def __extract_collscan(self, node: dict) -> list[float]:
        feats = []
        feats.extend(self.__encode_filter(node.get('filter')))
        feats.append(self.direction_map.get(node.get('direction', 'forward'), 0.0))
        return feats

    def __dim_collscan(self) -> int:
        return self.__dim_filter() + 1

    #endregion
    #region IXSCAN

    def __extract_ixscan(self, node: dict) -> list[float]:
        feats = []
        # Index name one-hot
        feats.extend(self.__one_hot(node.get('indexName', ''), self.index_name_vocab))
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

    def __dim_ixscan(self) -> int:
        return len(self.index_name_vocab) + 4 + 4 + 1

    #endregion
    #region FETCH

    def __extract_fetch(self, node: dict) -> list[float]:
        feats = []
        feats.extend(self.__encode_filter(node.get('filter')))
        return feats

    def __dim_fetch(self) -> int:
        return self.__dim_filter()

    #endregion
    #region SORT

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

    def __dim_sort(self) -> int:
        return 4

    #endregion
    #region LIMIT / SKIP

    def __extract_limit_skip(self, node: dict, stage: str) -> list[float]:
        feats = []
        amount = node.get('limitAmount', 0) if stage == 'LIMIT' else node.get('skipAmount', 0)
        feats.append(self.__log_normalize(amount))
        return feats

    def __dim_limit_skip(self) -> int:
        return 1

    #endregion
    #region PROJECTION

    def __extract_projection(self, node: dict) -> list[float]:
        feats = []
        transform = node.get('transformBy', {})
        feats.append(self.__log_normalize(len(transform)))
        return feats

    def __dim_projection(self) -> int:
        return 1

    #endregion
    #region GROUP

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

    def __dim_group(self) -> int:
        return 1 + len(self._GROUP_ACCUMULATORS) + 1  # num_accumulators + acc_vec + key_complexity

    #endregion
    #region EQ_LOOKUP

    def __extract_eq_lookup(self, node: dict) -> list[float]:
        feats = []
        # Foreign collection one-hot
        foreign_ns = node.get('foreignCollection', '')
        foreign_coll = foreign_ns.split('.')[-1] if '.' in foreign_ns else foreign_ns
        feats.extend(self.__one_hot(foreign_coll, self.collection_vocab))
        # Foreign collection stats
        feats.extend(self.__extract_global_features(foreign_coll))
        # Strategy
        strategy = node.get('strategy', '')
        strategies = ['NestedLoopJoin', 'IndexedLoopJoin', 'HashLookup']
        for s in strategies:
            feats.append(1.0 if strategy == s else 0.0)
        return feats

    def __dim_eq_lookup(self) -> int:
        return len(self.collection_vocab) + self.__dim_global_features() + 3

    #endregion
    #region MERGE / AND / OR

    def __extract_merge_ops(self, node: dict) -> list[float]:
        feats = []
        feats.extend(self.__encode_filter(node.get('filter')))
        return feats

    def __dim_merge_ops(self) -> int:
        return self.__dim_filter()

    #endregion
    #region Generic fallback

    def __extract_generic(self, node: dict) -> list[float]:
        feats = []
        feats.extend(self.__encode_filter(node.get('filter')))
        feats.append(self.direction_map.get(node.get('direction', 'forward'), 0.0))
        return feats

    def __dim_generic(self) -> int:
        return self.__dim_filter() + 1

    #endregion
    #region Global features

    def __extract_global_features(self, collection_name: str) -> np.ndarray:
        """
        Global context features for the collection:
        count, size, avgObjSize, storageSize, nindexes (all log-normalized).
        """
        stats = self.collection_stats.get(collection_name, {})
        feats = []
        for key in ['count', 'size', 'avgObjSize', 'storageSize', 'nindexes']:
            feats.append(self.__log_normalize(stats.get(key, 0)))
        return np.array(feats, dtype=np.float32)

    def __dim_global_features(self) -> int:
        return 5  # count, size, avgObjSize, storageSize, nindexes

    #endregion
    #region Encodigs

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

    def __dim_filter(self) -> int:
        return len(FILTER_OPS) + 11

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

    #endregion
    #region Helpers

    def __log_normalize(self, value: float) -> float:
        """Log-scale normalization for wide-range values."""
        return math.log1p(max(0, value))

    def __one_hot(self, value: str, vocab: set) -> np.ndarray:
        vocab_list = sorted(vocab)
        vec = np.zeros(len(vocab_list), dtype=np.float32)
        if value in vocab_list:
            vec[vocab_list.index(value)] = 1.0
        return vec

    #endregion
    #region Kinds

    def extract_query_kinds(self, query: MongoQuery) -> set[str]:
        """Returns the set of kinds that have to be in the database for the query to be executed."""
        collections = set[str](query.collection)
        if isinstance(query, MongoAggregateQuery):
            collections.update(self._extract_mongo_pipeline_collections(query.pipeline))

        return collections

    def _extract_mongo_pipeline_collections(self, pipeline: list[dict]) -> set[str]:
        collections = set[str]()

        def append_collection(name: str | None):
            if name is not None:
                collections.add(name)

        def visit_node(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in ('$lookup', '$graphLookup') and isinstance(value, dict):
                        append_collection(value.get('from'))
                        if 'pipeline' in value:
                            visit_node(value['pipeline'])
                        continue

                    if key == '$unionWith':
                        if isinstance(value, str):
                            append_collection(value)
                        elif isinstance(value, dict):
                            append_collection(value.get('coll'))
                            if 'pipeline' in value:
                                visit_node(value['pipeline'])
                        continue

                    visit_node(value)
            elif isinstance(node, list):
                for item in node:
                    visit_node(item)

        visit_node(pipeline)

        return collections

