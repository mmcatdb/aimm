"""
Feature extraction from PostgreSQL query plans.
Converts the JSON plan tree into vectorized inputs for neural units.
Each operator type has different features, handled by specific extractors.
Features are normalized and encoded appropriately (numeric, one-hot, boolean).
"""
import numpy as np
from collections import defaultdict

class FeatureExtractor:
    """
    Extracts and encodes features from query plan operators.
    Maintains vocabularies for categorical features across the dataset.
    """

    def __init__(self):
        # Vocabularies for one-hot encoding
        self.node_type_vocab = set()
        self.join_type_vocab = set()
        self.scan_type_vocab = set()  # e.g. Seq Scan, Index Scan, etc.
        self.scan_relation_vocab = set()
        self.sort_method_vocab = set()
        self.index_name_vocab = set()
        self.hash_algorithm_vocab = set()
        self.agg_strategy_vocab = set()
        self.agg_operator_vocab = set()
        self.sort_key_vocab = set()

        # Statistics for normalization (mean, std)
        self.numeric_stats = {}

    def build_vocabularies(self, plans: list[dict]):
        """
        Build vocabularies from training data for one-hot encoding.
        Also compute statistics for numeric feature normalization.
        """
        numeric_features = defaultdict(list)

        def traverse_plan(node):
            """Recursively traverse plan tree to collect features."""
            node_type = node.get('Node Type', '')
            self.node_type_vocab.add(node_type)

            if 'Scan' in node_type:
                self.scan_type_vocab.add(node_type)

            # NOTE: I realize 'Startup Cost' isn't here, but for some reason the model's performance
            #   completely craters if the feature is normalized.
            for key in ['Plan Width', 'Plan Rows', 'Plan Buffers', 'Estimated I/Os', 'Hash Buckets', 'Sort Space Used', 'Total Cost']:
                if key in node:
                    numeric_features[key].append(float(node[key]))

            # Collect categorical features
            if 'Join Type' in node:
                self.join_type_vocab.add(node['Join Type'])

            if 'Relation Name' in node:
                self.scan_relation_vocab.add(node['Relation Name'])

            if 'Sort Method' in node:
                self.sort_method_vocab.add(node['Sort Method'])

            if 'Sort Key' in node:
                # Keys can be a list, so convert to a stable string
                self.sort_key_vocab.add(str(node['Sort Key']))

            if 'Index Name' in node:
                self.index_name_vocab.add(node['Index Name'])

            if 'Hash Algorithm' in node:
                self.hash_algorithm_vocab.add(node['Hash Algorithm'])

            if 'Strategy' in node:
                self.agg_strategy_vocab.add(node['Strategy'])

            if 'Operator' in node:
                self.agg_operator_vocab.add(node['Operator'])

            # Recursively process children
            if 'Plans' in node:
                for child in node['Plans']:
                    traverse_plan(child)

        # Process all plans
        for plan in plans:
            traverse_plan(plan)

        # Compute normalization statistics
        for key, values in numeric_features.items():
            self.numeric_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values) + 1e-8   # avoid div by zero
            }

        print(f"Built vocabularies:")
        print(f"- Node types: {len(self.node_type_vocab)}")
        print(f"- Scan types: {len(self.scan_type_vocab)}")
        print(f"- Join types: {len(self.join_type_vocab)}")
        print(f"- Relations: {len(self.scan_relation_vocab)}")
        print(f"- Sort Keys: {len(self.sort_key_vocab)}")
        print(f"- Agg Operators: {len(self.agg_operator_vocab)}")

    def normalize_numeric(self, value: float, feature_name: str) -> float:
        """Normalize a numeric feature using z-score normalization."""
        if feature_name in self.numeric_stats:
            stats = self.numeric_stats[feature_name]
            return (value - stats['mean']) / stats['std']
        return value

    def encode_one_hot(self, value: str, vocabulary: set) -> np.ndarray:
        """Encode a categorical value as a one-hot vector."""
        vocab_list = sorted(list(vocabulary))
        vector = np.zeros(len(vocab_list))
        if value in vocab_list:
            idx = vocab_list.index(value)
            vector[idx] = 1.0
        return vector

    def extract_common_features(self, node: dict) -> np.ndarray:
        """
        Extract features common to all operator types.
        These appear in every neural unit's input.
        """
        features = []

        # Numeric features (normalized)
        features.append(self.normalize_numeric(node.get('Plan Width', 0), 'Plan Width'))
        features.append(self.normalize_numeric(node.get('Plan Rows', 0), 'Plan Rows'))
        features.append(self.normalize_numeric(node.get('Total Cost', 0), 'Total Cost'))
        features.append(self.normalize_numeric(node.get('Plan Buffers', 0), 'Plan Buffers'))
        features.append(self.normalize_numeric(node.get('Estimated I/Os', 0), 'Estimated I/Os'))
        features.append(self.normalize_numeric(node.get('Startup Cost', 0), 'Startup Cost'))

        return np.array(features)

    def extract_scan_features(self, node: dict) -> np.ndarray:
        """Extract features specific to scan operators."""
        features = []

        # Common features
        features.extend(self.extract_common_features(node))

        node_type = node.get('Node Type', '')
        scan_type_vec = self.encode_one_hot(node_type, self.scan_type_vocab)
        features.extend(scan_type_vec)

        # Relation name (one-hot)
        if 'Relation Name' in node:
            relation_vec = self.encode_one_hot(node['Relation Name'], self.scan_relation_vocab)
            features.extend(relation_vec)
        else:
            features.extend(np.zeros(len(self.scan_relation_vocab)))

        if 'Index Name' in node:
            index_vec = self.encode_one_hot(node['Index Name'], self.index_name_vocab)
            features.extend(index_vec)
        else:
            features.extend(np.zeros(len(self.index_name_vocab)))


        # Scan direction (bool) for index scans
        scan_direction = 1.0 if node.get('Scan Direction', 'Forward') == 'Forward' else 0.0
        features.append(scan_direction)

        # TODO: Attribute Mins/Medians/Maxs from Table 2???

        return np.array(features)

    def extract_join_features(self, node: dict) -> np.ndarray:
        """Extract features specific to join operators."""
        features = []

        features.extend(self.extract_common_features(node))

        # Join type (one-hot)
        if 'Join Type' in node:
            join_vec = self.encode_one_hot(node['Join Type'], self.join_type_vocab)
            features.extend(join_vec)
        else:
            features.extend(np.zeros(len(self.join_type_vocab)))

        # Parent relationship (if exists)
        parent_rel = node.get('Parent Relationship', 'Unknown')
        parent_vec = np.zeros(3)  # inner, outer, subquery
        if parent_rel == 'Inner':
            parent_vec[0] = 1.0
        elif parent_rel == 'Outer':
            parent_vec[1] = 1.0
        elif parent_rel == 'SubPlan': # 'Subquery' in paper
            parent_vec[2] = 1.0
        features.extend(parent_vec)

        # NOTE: Hash Buckets and Hash Algorithm are handled in
        # extract_hash_features, as 'Hash' is a separate node type
        # from 'Hash Join' (which is caught here).

        return np.array(features)

    def extract_hash_features(self, node: dict) -> np.ndarray:
        """Extract features specific to hash operators."""
        features = []

        # Common features
        features.extend(self.extract_common_features(node))

        # Hash Buckets (numeric)
        features.append(self.normalize_numeric(node.get('Hash Buckets', 0), 'Hash Buckets'))

        # Hash Algorithm (one-hot)
        if 'Hash Algorithm' in node:
            hash_vec = self.encode_one_hot(node['Hash Algorithm'], self.hash_algorithm_vocab)
            features.extend(hash_vec)
        else:
            features.extend(np.zeros(len(self.hash_algorithm_vocab)))

        return np.array(features)

    def extract_aggregate_features(self, node: dict) -> np.ndarray:
        """Extract features specific to aggregate operators."""
        features = []

        # Common features
        features.extend(self.extract_common_features(node))

        # Strategy (one-hot)
        if 'Strategy' in node:
            strategy_vec = self.encode_one_hot(node['Strategy'], self.agg_strategy_vocab)
            features.extend(strategy_vec)
        else:
            features.extend(np.zeros(len(self.agg_strategy_vocab)))

        # NOTE: This is 'Partial Mode' from paper (maps to 'Parallel Aware')... i hope
        partial_mode = 1.0 if node.get('Parallel Aware', False) else 0.0
        features.append(partial_mode)

        if 'Operator' in node:
            op_vec = self.encode_one_hot(node['Operator'], self.agg_operator_vocab)
            features.extend(op_vec)
        else:
            features.extend(np.zeros(len(self.agg_operator_vocab)))

        return np.array(features)


    def extract_sort_features(self, node: dict) -> np.ndarray:
        """Extract features specific to sort operators."""
        features = []

        # Common features
        features.extend(self.extract_common_features(node))

        # Sort method (one-hot)
        if 'Sort Method' in node:
            sort_vec = self.encode_one_hot(node['Sort Method'], self.sort_method_vocab)
            features.extend(sort_vec)
        else:
            features.extend(np.zeros(len(self.sort_method_vocab)))

        if 'Sort Key' in node:
            key_vec = self.encode_one_hot(str(node['Sort Key']), self.sort_key_vocab)
            features.extend(key_vec)
        else:
            features.extend(np.zeros(len(self.sort_key_vocab)))

        sort_space = node.get('Sort Space Used', 0)
        features.append(self.normalize_numeric(sort_space, 'Sort Space Used'))


        return np.array(features)

    def extract_features(self, node: dict) -> np.ndarray:
        """
        Extract features for any operator node based on its type.
        Routes to specific extraction methods.()
        """
        node_type = node.get('Node Type', '')

        # Scan operators
        if 'Scan' in node_type:
            return self.extract_scan_features(node)
        # Join operators
        elif 'Join' in node_type:
            return self.extract_join_features(node)
        # 'Hash' must come after 'Join', as 'Hash Join' should be routed to extract_join_features.
        elif 'Hash' in node_type:
            return self.extract_hash_features(node)
        # Aggregate operators
        elif 'Aggregate' in node_type or 'Group' in node_type:
            return self.extract_aggregate_features(node)
        # Sort operators
        elif 'Sort' in node_type:
            return self.extract_sort_features(node)
        else:
            return self.extract_common_features(node)

    def get_feature_dim(self, node_type: str) -> int:
        """Get the dimension of the feature vector for a given node type."""

        # NOTE: Everything except 'Node Type' isn't necessary to be here, but it gives better overview of the features.
        # - This is because `extract_common_features` doesn't care if 'Plan Width', 'Plan Rows', etc. are in the node or not.
        dummy_node = {
            'Node Type': node_type,
            'Plan Width': 0,
            'Plan Rows': 0,
            'Total Cost': 0,
            'Startup Cost': 0,
            'Plan Buffers': 0,
            'Estimated I/Os': 0
        }

        if 'Scan' in node_type:
            dummy_node['Node Type'] = next(iter(self.scan_type_vocab)) if self.scan_type_vocab else node_type
            dummy_node['Relation Name'] = 'dummy'
            dummy_node['Index Name'] = 'dummy_idx'
        elif 'Join' in node_type:
            dummy_node['Join Type'] = 'Inner'
            dummy_node['Parent Relationship'] = 'Inner'
        elif 'Hash' == node_type: # Be specific
            dummy_node['Hash Buckets'] = 0
            dummy_node['Hash Algorithm'] = 'dummy_alg'
        elif 'Aggregate' in node_type or 'Group' in node_type:
            dummy_node['Strategy'] = 'Plain'
            dummy_node['Operator'] = 'dummy_op'
        elif 'Sort' in node_type:
            dummy_node['Sort Method'] = 'quicksort'
            dummy_node['Sort Key'] = 'dummy_key'
            dummy_node['Sort Space Used'] = 0

        features = self.extract_features(dummy_node)
        return len(features)
