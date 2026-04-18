from typing_extensions import override
import numpy as np
from typing import Any
from collections import defaultdict
import re
from core.utils import EPSILON
from ..feature_extractor import PlanNode, BaseFeatureExtractor

class FeatureExtractor(BaseFeatureExtractor):
    """
    Feature extraction from Neo4j query plans.
    Converts the Neo4j plan tree into vectorized inputs for neural units.
    Features are normalized and encoded appropriately (numeric, one-hot, boolean).

    Extracts and encodes features from Neo4j query plan operators.
    Maintains vocabularies for categorical features across the dataset.
    """

    #region Setup

    def __init__(self):
        self.identifiers_vocab = set()

        # Statistics for normalization
        # Changed from lambda to dict to enable pickling
        self.stats = defaultdict(self.__default_stats)

    def __default_stats(self):
        """Default statistics dictionary for normalization."""
        return {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'count': 0}

    @override
    def extend_vocabularies(self, plans: list[dict], global_stats: dict):
        all_numeric_features = defaultdict(list)

        def traverse(node):
            # Extract identifiers
            identifiers = node.get('identifiers', [])
            for ident in identifiers:
                self.identifiers_vocab.add(ident)

            # Extract numeric features
            args = node.get('args', {})

            # EstimatedRows
            estimated_rows = args.get('EstimatedRows', 0)
            if estimated_rows is not None:
                all_numeric_features['EstimatedRows'].append(float(estimated_rows))

            # Number of identifiers
            all_numeric_features['num_identifiers'].append(len(identifiers))

            # Number of children
            children = node.get('children', [])
            all_numeric_features['num_children'].append(len(children))

            # Parse details
            details = args.get('Details', '')
            if details:
                all_numeric_features['details_length'].append(len(details))
                details_info = self.__parse_details(details)
                all_numeric_features['num_variables'].append(details_info['num_variables'])

            # Recursively traverse children
            for child in children:
                traverse(child)

        # Traverse all plans
        for plan in plans:
            traverse(plan)

        # Compute statistics for numeric features
        for feature_name, values in all_numeric_features.items():
            if values:
                self.stats[feature_name]['min'] = float(np.min(values))
                self.stats[feature_name]['max'] = float(np.max(values))
                self.stats[feature_name]['mean'] = float(np.mean(values))
                self.stats[feature_name]['count'] = len(values)

    #endregion

    @override
    def extract_plan(self, plan: dict) -> PlanNode:
        node_type = self.get_node_type(plan)
        features = self.__extract_node_features(plan)
        node = PlanNode(node_type, features, None)

        for child in self.get_node_children(plan):
            child_node = self.extract_plan(child)
            node.add_child(child_node)

        return node

    #region Features

    def __extract_node_features(self, node: dict) -> np.ndarray:
        """
        Extract and encode features from a single operator node.
        Args:
            node: A Neo4j query plan operator node (dictionary)
        Returns:
            Feature vector as numpy array
        """
        features = []

        # Arguments
        args = node.get('args', {})

        # EstimatedRows (normalized)
        estimated_rows = args.get('EstimatedRows', 0)
        if estimated_rows is None:
            estimated_rows = 0
        estimated_rows_norm = self.__normalize_numeric(float(estimated_rows), 'EstimatedRows')
        features.append(np.array([estimated_rows_norm], dtype=np.float32))


        # Identifiers (multi-hot encoded)
        identifiers = node.get('identifiers', [])
        if self.identifiers_vocab:
            identifiers_vec = self._multi_hot(identifiers, self.identifiers_vocab)
            features.append(identifiers_vec)

        # Number of identifiers (normalized)
        num_identifiers = len(identifiers)
        num_identifiers_norm = self.__normalize_numeric(float(num_identifiers), 'num_identifiers')
        features.append(np.array([num_identifiers_norm], dtype=np.float32))

        # Number of children (normalized)
        children = node.get('children', [])
        num_children = len(children)
        num_children_norm = self.__normalize_numeric(float(num_children), 'num_children')
        features.append(np.array([num_children_norm], dtype=np.float32))

        # Details parsing (boolean features)
        details = args.get('Details', '')
        details_info = self.__parse_details(details)

        details_vec = np.array([
            float(details_info['has_predicate']),
            float(details_info['has_relationship']),
            float(details_info['has_property_access']),
            float(details_info['has_aggregation']),
            float(details_info['has_comparison']),
        ], dtype=np.float32)
        features.append(details_vec)

        # Details length (normalized)
        if details:
            details_length_norm = self.__normalize_numeric(float(len(details)), 'details_length')
        else:
            details_length_norm = 0.0
        features.append(np.array([details_length_norm], dtype=np.float32))

        # Number of variables in details (normalized)
        num_vars_norm = self.__normalize_numeric(float(details_info['num_variables']), 'num_variables')
        features.append(np.array([num_vars_norm], dtype=np.float32))

        # Concatenate all features
        feature_vector = np.concatenate(features)

        return feature_vector

    def __parse_details(self, details: str) -> dict[str, Any]:
        """
        Parse the 'Details' field from Neo4j operators.
        Extract useful information like predicates, relationship types, etc.
        """
        info = {
            'has_predicate': False,
            'has_relationship': False,
            'has_property_access': False,
            'has_aggregation': False,
            'has_comparison': False,
            'num_variables': 0
        }

        if not details:
            return info

        # Check for predicates (=, <, >, <=, >=, <>)
        if re.search(r'[=<>!]+', details):
            info['has_predicate'] = True
            info['has_comparison'] = True

        # Check for relationship patterns
        if re.search(r'\[:[A-Z_]+\]', details):
            info['has_relationship'] = True

        # Check for property access (e.g., n.name, c.acctbal)
        if re.search(r'\w+\.\w+', details):
            info['has_property_access'] = True

        # Check for aggregation functions
        if re.search(r'(count|sum|avg|min|max|collect)\s*\(', details, re.IGNORECASE):
            info['has_aggregation'] = True

        # Count number of variables (simple heuristic: single letters followed by non-letter or dots)
        variables = re.findall(r'\b[a-z]\b|\b[a-z](?=\.)', details)
        info['num_variables'] = len(set(variables))

        return info

    def __normalize_numeric(self, value: float, feature_name: str) -> float:
        """
        Normalize a numeric feature using min-max normalization.
        Returns value in [0, 1] range.
        """
        stats = self.stats[feature_name]
        min_val = stats['min']
        max_val = stats['max']

        if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val < min_val:
            return 0.0

        if max_val == min_val:
            return 0.5  # If all values are the same, return middle value

        # Min-max normalization with log scaling for large ranges
        if max_val > 1000 or min_val < -1000:
            # Use log scaling for large values
            if value > 0:
                value = np.log1p(value)
                min_val = np.log1p(max(0, min_val))
                max_val = np.log1p(max_val)

        normalized = (value - min_val) / (max_val - min_val + EPSILON)
        return float(np.clip(normalized, 0, 1))

    #endregion

    @staticmethod
    @override
    def get_node_type(node: dict) -> str:
        return node.get('operatorType', 'Unknown').replace('@neo4j', '')

    @staticmethod
    @override
    def get_node_children(node: dict) -> list[dict]:
        return node.get('children', [])
