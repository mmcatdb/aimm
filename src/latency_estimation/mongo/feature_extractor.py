"""
Feature extraction from MongoDB query plans.

Handles both Classic Engine (explainVersion "1") and SBE (explainVersion "2") plans.
For SBE plans, we use the queryPlan JSON tree, not the slotBasedPlan text.

Key differences from PostgreSQL:
- No pre-execution cardinality estimates (no Plan Rows / Plan Width)
- Filter predicates must be encoded to help the network learn selectivity
- Collection stats serve as the baseline for data volume estimation
"""
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict
import math


# All known MongoDB execution stages
KNOWN_STAGES = [
    "COLLSCAN", "IXSCAN", "FETCH", "SORT", "SORT_MERGE",
    "LIMIT", "SKIP", "PROJECTION_SIMPLE", "PROJECTION_DEFAULT",
    "PROJECTION_COVERED", "AND_HASH", "AND_SORTED", "OR",
    "COUNT_SCAN", "TEXT_MATCH", "TEXT_OR", "GEO_NEAR_2D",
    "GEO_NEAR_2DSPHERE", "SHARDING_FILTER", "SUBPLAN",
    "EQ_LOOKUP", "GROUP", "UNWIND", "EOF",
    # SBE-specific
    "NLJ", "HJ",  # nested loop join, hash join
]

# MongoDB filter operators
FILTER_OPS = [
    "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
    "$in", "$nin", "$regex", "$exists", "$type",
    "$all", "$elemMatch", "$size",
    "$and", "$or", "$nor", "$not",
]


class FeatureExtractor:
    """
    Extracts and encodes features from MongoDB query plan nodes.
    """

    def __init__(self):
        self.stage_vocab = set()
        self.collection_vocab = set()
        self.index_name_vocab = set()
        self.direction_map = {"forward": 1.0, "backward": -1.0}
        self.numeric_stats = {}
        # Collection stats cache (loaded before training/inference)
        self.coll_stats: Dict[str, Dict] = {}
        # Global features dimension (computed after build_vocabularies)
        self._global_dim = None

    def set_collection_stats(self, stats: Dict[str, Dict]):
        """Set collection statistics for feature extraction."""
        self.coll_stats = stats

    def build_vocabularies(self, plans: List[Dict]):
        """
        Build vocabularies by scanning all plan trees.
        plans: list of plan root nodes (the winningPlan / queryPlan dicts).
        """
        numeric_features = defaultdict(list)

        def traverse(node):
            stage = node.get("stage", "UNKNOWN")
            self.stage_vocab.add(stage)

            # Collect collection names
            ns = node.get("namespace", "")
            if ns:
                coll = ns.split(".")[-1] if "." in ns else ns
                self.collection_vocab.add(coll)

            if "indexName" in node:
                self.index_name_vocab.add(node["indexName"])

            # Numeric features for normalization
            for key in ["limitAmount", "skipAmount", "memLimit"]:
                if key in node:
                    numeric_features[key].append(float(node[key]))

            # Recurse children
            if "inputStage" in node:
                traverse(node["inputStage"])
            if "inputStages" in node:
                for child in node["inputStages"]:
                    traverse(child)

        for plan in plans:
            traverse(plan)

        # Add collection names from stats if not yet seen
        for coll_name in self.coll_stats:
            self.collection_vocab.add(coll_name)

        # Compute normalization stats
        for key, values in numeric_features.items():
            self.numeric_stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values) + 1e-8,
            }

        # Add collection-stats normalization
        for stat_key in ["count", "size", "avgObjSize", "storageSize", "nindexes"]:
            vals = [s[stat_key] for s in self.coll_stats.values() if stat_key in s]
            if vals:
                self.numeric_stats[f"coll_{stat_key}"] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals) + 1e-8,
                }

        self._global_dim = None  # reset cache

        print(f"Built vocabularies:")
        print(f"  Stages: {sorted(self.stage_vocab)}")
        print(f"  Collections: {sorted(self.collection_vocab)}")
        print(f"  Indexes: {sorted(self.index_name_vocab)}")

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _normalize(self, value: float, key: str) -> float:
        if key in self.numeric_stats:
            s = self.numeric_stats[key]
            return (value - s["mean"]) / s["std"]
        return value

    def _log_normalize(self, value: float) -> float:
        """Log-scale normalization for wide-range values."""
        return math.log1p(max(0, value))

    def _one_hot(self, value: str, vocab: set) -> np.ndarray:
        vocab_list = sorted(vocab)
        vec = np.zeros(len(vocab_list), dtype=np.float32)
        if value in vocab_list:
            vec[vocab_list.index(value)] = 1.0
        return vec

    # ------------------------------------------------------------------
    # Global / collection features
    # ------------------------------------------------------------------

    def _extract_global_features(self, collection_name: str) -> np.ndarray:
        """
        Global context features for the collection:
        count, size, avgObjSize, storageSize, nindexes (all log-normalized).
        """
        stats = self.coll_stats.get(collection_name, {})
        feats = []
        for key in ["count", "size", "avgObjSize", "storageSize", "nindexes"]:
            feats.append(self._log_normalize(stats.get(key, 0)))
        return np.array(feats, dtype=np.float32)

    def get_global_dim(self) -> int:
        return 5  # count, size, avgObjSize, storageSize, nindexes

    # ------------------------------------------------------------------
    # Filter / predicate encoding
    # ------------------------------------------------------------------

    def _encode_filter(self, filter_doc: Optional[Dict]) -> np.ndarray:
        """
        Encode a filter predicate into a fixed-size feature vector.
        Returns: operator presence vector + tree depth + num predicates + has_regex
                 + numeric value summary (min, max, mean, count, range_width)
                 + num_fields + is_equality + num_in_values
        """
        op_vec = np.zeros(len(FILTER_OPS), dtype=np.float32)
        stats = {"depth": 0, "num_predicates": 0, "has_regex": 0}
        numeric_values = []
        num_fields = 0
        is_equality = 0.0
        num_in_values = 0

        def walk(doc, depth=0):
            nonlocal num_fields, is_equality, num_in_values
            if not isinstance(doc, dict):
                return
            stats["depth"] = max(stats["depth"], depth)
            for k, v in doc.items():
                if k in FILTER_OPS:
                    idx = FILTER_OPS.index(k)
                    op_vec[idx] = 1.0
                    stats["num_predicates"] += 1
                    if k == "$regex":
                        stats["has_regex"] = 1
                    if k == "$eq":
                        is_equality = 1.0
                    # Extract numeric values from comparison operators
                    if k in ("$gt", "$gte", "$lt", "$lte", "$eq", "$ne"):
                        if isinstance(v, (int, float)):
                            numeric_values.append(float(v))
                        elif hasattr(v, 'timestamp'):
                            # datetime -> epoch seconds (normalized)
                            numeric_values.append(v.timestamp() / 1e9)
                    if k == "$in" and isinstance(v, list):
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
                elif not k.startswith("$"):
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
            walk(filter_doc)

        # Numeric value summary features
        if numeric_values:
            nv = np.array(numeric_values)
            val_min = self._log_normalize(abs(float(np.min(nv))))
            val_max = self._log_normalize(abs(float(np.max(nv))))
            val_mean = self._log_normalize(abs(float(np.mean(nv))))
            val_count = self._log_normalize(len(nv))
            val_range = self._log_normalize(abs(float(np.max(nv) - np.min(nv))))
        else:
            val_min = val_max = val_mean = val_count = val_range = 0.0

        extra = np.array([
            stats["depth"],
            self._log_normalize(stats["num_predicates"]),
            stats["has_regex"],
            val_min,
            val_max,
            val_mean,
            val_count,
            val_range,
            self._log_normalize(num_fields),
            is_equality,
            self._log_normalize(num_in_values),
        ], dtype=np.float32)
        return np.concatenate([op_vec, extra])

    def get_filter_dim(self) -> int:
        return len(FILTER_OPS) + 11 

    # ------------------------------------------------------------------
    # Index bounds encoding
    # ------------------------------------------------------------------

    def _encode_index_bounds(self, bounds: Optional[Dict]) -> np.ndarray:
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
                        # e.g. "[1, 1]" vs "[1, 100]"
                        parts = iv.strip("[]").split(",")
                        if len(parts) == 2:
                            lo, hi = parts[0].strip(), parts[1].strip()
                            if lo != hi:
                                is_point = 0.0
                                is_range = 1.0

        return np.array([
            self._log_normalize(num_fields),
            self._log_normalize(total_intervals),
            is_point,
            is_range,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Per-stage feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, node: Dict, collection_name: str = "") -> np.ndarray:
        """
        Extract feature vector for a single plan node.
        """
        stage = node.get("stage", "UNKNOWN")
        features = []

        # 1. Stage one-hot
        features.extend(self._one_hot(stage, self.stage_vocab))

        # 2. Global collection features
        features.extend(self._extract_global_features(collection_name))

        # 3. Stage-specific features
        if stage == "COLLSCAN":
            features.extend(self._extract_collscan(node))
        elif stage in ("IXSCAN", "EXPRESS_IXSCAN"):
            features.extend(self._extract_ixscan(node))
        elif stage == "FETCH":
            features.extend(self._extract_fetch(node))
        elif stage == "SORT":
            features.extend(self._extract_sort(node))
        elif stage in ("LIMIT", "SKIP"):
            features.extend(self._extract_limit_skip(node, stage))
        elif stage in ("PROJECTION_SIMPLE", "PROJECTION_DEFAULT", "PROJECTION_COVERED"):
            features.extend(self._extract_projection(node))
        elif stage == "GROUP":
            features.extend(self._extract_group(node))
        elif stage == "EQ_LOOKUP":
            features.extend(self._extract_eq_lookup(node))
        elif stage in ("AND_HASH", "AND_SORTED", "OR", "SORT_MERGE"):
            features.extend(self._extract_merge_ops(node))
        else:
            # Generic fallback: filter + direction
            features.extend(self._extract_generic(node))

        return np.array(features, dtype=np.float32)

    # --- COLLSCAN ---
    def _extract_collscan(self, node: Dict) -> List[float]:
        feats = []
        feats.extend(self._encode_filter(node.get("filter")))
        feats.append(self.direction_map.get(node.get("direction", "forward"), 0.0))
        return feats

    def _collscan_dim(self) -> int:
        return self.get_filter_dim() + 1

    # --- IXSCAN ---
    def _extract_ixscan(self, node: Dict) -> List[float]:
        feats = []
        # Index name one-hot
        feats.extend(self._one_hot(node.get("indexName", ""), self.index_name_vocab))
        # isMultiKey
        feats.append(1.0 if node.get("isMultiKey", False) else 0.0)
        # isUnique
        feats.append(1.0 if node.get("isUnique", False) else 0.0)
        # isSparse
        feats.append(1.0 if node.get("isSparse", False) else 0.0)
        # direction
        feats.append(self.direction_map.get(node.get("direction", "forward"), 0.0))
        # Index bounds encoding
        feats.extend(self._encode_index_bounds(node.get("indexBounds")))
        # Number of key fields in the index
        key_pattern = node.get("keyPattern", {})
        feats.append(self._log_normalize(len(key_pattern)))
        return feats

    def _ixscan_dim(self) -> int:
        return len(self.index_name_vocab) + 4 + 4 + 1

    # --- FETCH ---
    def _extract_fetch(self, node: Dict) -> List[float]:
        feats = []
        feats.extend(self._encode_filter(node.get("filter")))
        return feats

    def _fetch_dim(self) -> int:
        return self.get_filter_dim()

    # --- SORT ---
    def _extract_sort(self, node: Dict) -> List[float]:
        feats = []
        sort_pattern = node.get("sortPattern", {})
        feats.append(self._log_normalize(len(sort_pattern)))
        mem_limit = node.get("memLimit", 104857600)
        feats.append(self._log_normalize(mem_limit))
        feats.append(self._log_normalize(node.get("limitAmount", 0)))
        # sort type
        sort_type = node.get("type", "simple")
        feats.append(1.0 if sort_type == "simple" else 0.0)
        return feats

    def _sort_dim(self) -> int:
        return 4

    # --- LIMIT / SKIP ---
    def _extract_limit_skip(self, node: Dict, stage: str) -> List[float]:
        feats = []
        amount = node.get("limitAmount", 0) if stage == "LIMIT" else node.get("skipAmount", 0)
        feats.append(self._log_normalize(amount))
        return feats

    def _limit_skip_dim(self) -> int:
        return 1

    # --- PROJECTION ---
    def _extract_projection(self, node: Dict) -> List[float]:
        feats = []
        transform = node.get("transformBy", {})
        feats.append(self._log_normalize(len(transform)))
        return feats

    def _projection_dim(self) -> int:
        return 1

    # --- GROUP ---
    # Accumulator operators we track
    _GROUP_ACCUMULATORS = ["$sum", "$avg", "$min", "$max", "$first", "$last",
                           "$push", "$addToSet", "$count", "$stdDevPop", "$stdDevSamp"]

    def _extract_group(self, node: Dict) -> List[float]:
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
        slots = node.get("slots", "")
        if isinstance(slots, str):
            num_accumulators = slots.count("$")

        # Count accumulator types if we can find accumulator specs
        accumulators = node.get("accumulators", [])
        if isinstance(accumulators, list):
            for acc in accumulators:
                if isinstance(acc, dict):
                    for op in acc.values():
                        if isinstance(op, str) and op in self._GROUP_ACCUMULATORS:
                            idx = self._GROUP_ACCUMULATORS.index(op)
                            acc_vec[idx] = 1.0
                            num_accumulators += 1

        feats.append(self._log_normalize(num_accumulators))
        feats.extend(acc_vec.tolist())

        # Group key complexity (1 = simple, >1 = compound)
        group_by = node.get("groupBy", node.get("_id", ""))
        if isinstance(group_by, dict):
            feats.append(self._log_normalize(len(group_by)))
        else:
            feats.append(0.0)

        return feats

    def _group_dim(self) -> int:
        return 1 + len(self._GROUP_ACCUMULATORS) + 1  # num_accumulators + acc_vec + key_complexity

    # --- EQ_LOOKUP ---
    def _extract_eq_lookup(self, node: Dict) -> List[float]:
        feats = []
        # Foreign collection one-hot
        foreign_ns = node.get("foreignCollection", "")
        foreign_coll = foreign_ns.split(".")[-1] if "." in foreign_ns else foreign_ns
        feats.extend(self._one_hot(foreign_coll, self.collection_vocab))
        # Foreign collection stats
        feats.extend(self._extract_global_features(foreign_coll))
        # Strategy
        strategy = node.get("strategy", "")
        strategies = ["NestedLoopJoin", "IndexedLoopJoin", "HashLookup"]
        for s in strategies:
            feats.append(1.0 if strategy == s else 0.0)
        return feats

    def _eq_lookup_dim(self) -> int:
        return len(self.collection_vocab) + self.get_global_dim() + 3

    # --- Merge / AND / OR ---
    def _extract_merge_ops(self, node: Dict) -> List[float]:
        feats = []
        feats.extend(self._encode_filter(node.get("filter")))
        return feats

    def _merge_ops_dim(self) -> int:
        return self.get_filter_dim()

    # --- Generic fallback ---
    def _extract_generic(self, node: Dict) -> List[float]:
        feats = []
        feats.extend(self._encode_filter(node.get("filter")))
        feats.append(self.direction_map.get(node.get("direction", "forward"), 0.0))
        return feats

    def _generic_dim(self) -> int:
        return self.get_filter_dim() + 1

    # ------------------------------------------------------------------
    # Feature dimension computation
    # ------------------------------------------------------------------

    def get_feature_dim(self, stage: str) -> int:
        """Get total feature dimension for a given stage type."""
        base_dim = len(self.stage_vocab) + self.get_global_dim()

        if stage == "COLLSCAN":
            return base_dim + self._collscan_dim()
        elif stage in ("IXSCAN", "EXPRESS_IXSCAN"):
            return base_dim + self._ixscan_dim()
        elif stage == "FETCH":
            return base_dim + self._fetch_dim()
        elif stage == "SORT":
            return base_dim + self._sort_dim()
        elif stage in ("LIMIT", "SKIP"):
            return base_dim + self._limit_skip_dim()
        elif stage in ("PROJECTION_SIMPLE", "PROJECTION_DEFAULT", "PROJECTION_COVERED"):
            return base_dim + self._projection_dim()
        elif stage == "GROUP":
            return base_dim + self._group_dim()
        elif stage == "EQ_LOOKUP":
            return base_dim + self._eq_lookup_dim()
        elif stage in ("AND_HASH", "AND_SORTED", "OR", "SORT_MERGE"):
            return base_dim + self._merge_ops_dim()
        else:
            return base_dim + self._generic_dim()
