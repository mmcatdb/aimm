import torch
import torch.nn as nn
from neural_units import create_neural_unit, NeuralUnit
from feature_extractor import FeatureExtractor

class PlanStructuredNetwork(nn.Module):
    """
    Plan-structured neural network for Neo4j query performance prediction.

    Dynamically assembles neural units to match query plan structure.
    """

    def __init__(self, feature_extractor: FeatureExtractor,
                 hidden_dim: int = 128, num_layers: int = 5,
                 data_vec_dim: int = 32):
        """
        Args:
            feature_extractor: FeatureExtractor instance for converting operators to features
            hidden_dim: Hidden layer size for neural units
            num_layers: Number of hidden layers per neural unit
            data_vec_dim: Size of data vectors passed between operators
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.data_vec_dim = data_vec_dim

        # Dictionary to store neural units for each operator type
        # Key: (operator_type, num_children)
        # Value: NeuralUnit
        self.units = nn.ModuleDict()

        # Track which operator types we've seen
        self.operator_types = set()

    def _get_unit_key(self, operator_type: str, num_children: int) -> str:
        """
        Generate a unique key for a neural unit based on operator type and number of children.

        Args:
            operator_type: Neo4j operator type
            num_children: Number of child operators

        Returns:
            Unique key string
        """
        # Remove @neo4j suffix if present
        operator_type = operator_type.replace('@neo4j', '')
        return f'{operator_type}_{num_children}'

    def _ensure_unit_exists(self, operator_type: str, num_children: int, feature_dim: int) -> str:
        """
        Ensure a neural unit exists for the given operator configuration.
        Creates it if it doesn't exist.

        Args:
            operator_type: Neo4j operator type
            num_children: Number of child operators
            feature_dim: Dimension of operator's feature vector

        Returns:
            Key for accessing the unit
        """
        operator_type = operator_type.replace('@neo4j', '')
        key = self._get_unit_key(operator_type, num_children)

        if key not in self.units:
            # Create new neural unit
            unit = create_neural_unit(
                operator_type=operator_type,
                input_dim=feature_dim,
                num_children=num_children,
                data_vec_dim=self.data_vec_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers
            )
            self.units[key] = unit
            self.operator_types.add(operator_type)

        return key

    def _process_node(self, node: dict, cache: dict | None = None) -> dict[str, torch.Tensor]:
        """
        Recursively process a query plan node.

        Args:
            node: Neo4j query plan node
            cache: Optional cache for node outputs (for batch processing)

        Returns:
            Dictionary with 'data' and optionally 'latency' tensors
        """
        # Generate a unique ID for this node (for caching)
        node_id = id(node)

        if cache is not None and node_id in cache:
            return cache[node_id]

        # Extract operator features
        operator_type = node.get('operatorType', 'Unknown').replace('@neo4j', '')
        operator_features = self.feature_extractor.extract_features(node)
        operator_features_tensor = torch.tensor(operator_features, dtype=torch.float32)

        # Process children recursively
        children = node.get('children', [])
        num_children = len(children)

        child_outputs = []
        for child in children:
            child_output = self._process_node(child, cache)
            # Extract data vector from child
            child_outputs.append(child_output['data'])

        # Concatenate operator features with children data vectors
        if child_outputs:
            # Stack child data vectors: [num_children, data_vec_dim]
            child_data = torch.stack(child_outputs)
            # Flatten to [num_children * data_vec_dim]
            child_data_flat = child_data.flatten()
            # Concatenate with operator features
            input_tensor = torch.cat([operator_features_tensor, child_data_flat])
        else:
            input_tensor = operator_features_tensor

        # Add batch dimension: [1, input_dim]
        input_tensor = input_tensor.unsqueeze(0)

        # Get or create neural unit
        feature_dim = len(operator_features)
        unit_key = self._ensure_unit_exists(operator_type, num_children, feature_dim)
        unit = self.units[unit_key]

        # Forward pass through neural unit
        output = unit(input_tensor)

        # Cache result
        if cache is not None:
            cache[node_id] = output

        return output

    def forward(self, plan: dict) -> torch.Tensor:
        """
        Forward pass through the plan-structured network.

        Args:
            plan: Neo4j query plan (root node)

        Returns:
            Predicted latency as tensor [1, 1]
        """
        # Process the entire plan tree
        output = self._process_node(plan)

        # Root should have predicted latency
        if 'latency' not in output:
            raise ValueError(f'Root operator {plan.get("operatorType")} did not predict latency. Make sure root is ProduceResults.')

        return output['latency']

    def forward_batch(self, plans: list[dict]) -> torch.Tensor:
        """
        Process a batch of query plans.

        Note: For Neo4j, we process plans individually since they have
        different structures. Future optimization could group by structure.

        Args:
            plans: List of Neo4j query plans

        Returns:
            Tensor of predicted latencies [batch_size, 1]
        """
        predictions = []

        for plan in plans:
            pred = self.forward(plan)
            predictions.append(pred)

        # Stack into batch: [batch_size, 1]
        return torch.cat(predictions, dim=0)

    def initialize_units_from_plans(self, plans: list[dict]):
        """
        Pre-initialize neural units by scanning through all plans.
        This ensures all necessary units are created before training.

        Args:
            plans: List of Neo4j query plans
        """
        print('Initializing neural units from plans...')

        def scan_node(node):
            operator_type = node.get('operatorType', 'Unknown').replace('@neo4j', '')
            children = node.get('children', [])
            num_children = len(children)

            # Extract features to get dimension
            features = self.feature_extractor.extract_features(node)
            feature_dim = len(features)

            # Ensure unit exists
            self._ensure_unit_exists(operator_type, num_children, feature_dim)

            # Recursively scan children
            for child in children:
                scan_node(child)

        # Scan all plans
        for plan in plans:
            scan_node(plan)

        print(f'  Created {len(self.units)} unique neural units')
        print(f'  Operator types: {sorted(self.operator_types)}')

    def get_operator_info(self) -> dict:
        """
        Get information about operator types for model saving/loading.

        Returns:
            Dictionary with operator configuration
        """
        return {
            'operator_types': list(self.operator_types),
            'unit_keys': list(self.units.keys()),
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'data_vec_dim': self.data_vec_dim
        }

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
