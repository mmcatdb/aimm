"""
- Assembles neural units into trees.
- Creates a neural network with structure isomorphic to a query plan.
- Each operator in the plan is replaced by its corresponding neural unit.
- The network recursively computes outputs bottom-up through the tree.
"""
import torch
import torch.nn as nn
from typing import Any
from neural_units import (
    NeuralUnit, GenericUnit
)
from feature_extractor import FeatureExtractor

class PlanStructuredNetwork(nn.Module):
    """
    Plan-structured neural network for query performance prediction.

    Maintains a library of neural units (one per operator type).
    Dynamically assembles them into trees matching query plans.
    """

    def __init__(self, feature_extractor: FeatureExtractor,
                 hidden_dim: int = 128, num_layers: int = 5,
                 data_vec_dim: int = 32):
        """
        Args:
            feature_extractor: Feature extractor with built vocabularies
            hidden_dim: Number of neurons in hidden layers
            num_layers: Number of hidden layers per unit
            data_vec_dim: Dimension of data output vectors
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.data_vec_dim = data_vec_dim

        # Create neural units for each operator type
        self.units = nn.ModuleDict()

        # Store operator info for model saving/loading
        self.operator_info = {}

    def _normalize_node_type(self, node_type: str) -> str:
        """
        Normalize node type for neural unit lookup.
        All scan types (Seq Scan, Index Scan, etc.) are unified to 'Scan'
        """
        if 'Scan' in node_type:
            return 'Scan'
        return node_type

    def get_operator_info(self) -> dict[str, dict[str, int]]:
        """
        Get information about all initialized operators.
        Used for model saving/loading.

        Returns:
            Dictionary mapping operator type to its info (feature_dim, num_children)
        """
        return self.operator_info.copy()

    def initialize_units_from_plans(self, plans: list[dict]):
        """
        Initialize neural units by analyzing all operator types and their *number of children* in the training plans.

        Args:
            plans: List of query plan dictionaries
        """
        # Collect operator types with their typical number of children
        operator_info_pairs = set()

        def collect_operator_info(node):
            """Recursively collect operator types and their children counts."""
            node_type = node.get('Node Type', '')
            # Normalize scan types to 'Scan' for unified neural unit
            normalized_type = self._normalize_node_type(node_type)
            num_children = len(node.get('Plans', []))

            operator_info_pairs.add((normalized_type, num_children))

            if 'Plans' in node:
                for child in node['Plans']:
                    collect_operator_info(child)

        # Collect all operator types
        for plan in plans:
            collect_operator_info(plan)

        # MODIFIED: Sort and print the pairs
        print(f'\nFound {len(operator_info_pairs)} unique operator type/children combinations:')
        sorted_pairs = sorted(list(operator_info_pairs))
        for op_type, num_children in sorted_pairs:
            print(f'  - {op_type} (children: {num_children})')

        # Create neural units for each (type, children) pair
        for node_type, num_children in sorted_pairs:

            feature_dim = self.feature_extractor.get_feature_dim(node_type)

            op_key = f'{node_type}_{num_children}'
            self.operator_info[op_key] = {
                'node_type': node_type,
                'feature_dim': feature_dim,
                'num_children': num_children
            }

            print(f'  Creating unit for {node_type} ({num_children} children): feature_dim={feature_dim}')

            # Create the neural unit
            self._create_unit(node_type, feature_dim, num_children)

        print(f'\nInitialized {len(self.units)} neural units')
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total model parameters: {total_params:,}')

    def initialize_units_from_operator_info(self, operator_info: dict[str, dict[str, Any]]):
        """
        Initialize neural units from saved operator information.
        Used when loading a saved model.

        Args:
            operator_info: Dictionary mapping operator type to {feature_dim, num_children}
        """
        print(f'\nInitializing {len(operator_info)} neural units from saved operator info...')

        for op_key, info in operator_info.items():
            node_type = info['node_type']
            feature_dim = info['feature_dim']
            num_children = info['num_children']

            print(f'  Creating unit for {node_type} ({num_children} children): feature_dim={feature_dim}')
            self._create_unit(node_type, feature_dim, num_children)

        self.operator_info = operator_info.copy()

        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total model parameters: {total_params:,}')

    def _create_unit(self, node_type: str, feature_dim: int, num_children: int = 0) -> nn.Module:
        """
        Create a neural unit for a specific operator type.
        """

        op_key = f'{node_type}_{num_children}'

        if op_key in self.units:
            return self.units[op_key]

        unit = GenericUnit(
            input_dim=feature_dim,
            num_children=num_children,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            data_vec_dim=self.data_vec_dim
        )

        self.units[op_key] = unit
        return unit

    def get_unit(self, node_type: str, num_children: int) -> nn.Module:
        """
        Get existing neural unit for (operator_type, num_children) pair.
        """
        op_key = f'{node_type}_{num_children}'

        if op_key not in self.units:
            # Potential fallback: could try to find the closest num_children.
            # For now, we'll just error out.
            raise ValueError(f'No neural unit found for operator type/children combination: {op_key}. Available types: {list(self.units.keys())}')
        return self.units[op_key]

    def process_node(self, node: dict, cache: dict | None = None) -> torch.Tensor:
        """
        Recursively process a query plan node.

        Args:
            node: Query plan node dictionary
            cache: Cache for storing intermediate results (optimization)

        Returns:
            Output tensor [1 + data_vec_dim] containing latency and data vector
        """
        if cache is None:
            cache = {}

        # Check cache (information sharing optimization)
        node_id = id(node)
        if node_id in cache:
            return cache[node_id]

        node_type = node.get('Node Type', '')
        normalized_type = self._normalize_node_type(node_type)


        # Process children (if any)
        children_outputs = []
        if 'Plans' in node:
            for child in node['Plans']:
                child_output = self.process_node(child, cache)
                children_outputs.append(child_output)

        num_children = len(children_outputs)

        node_features = self.feature_extractor.extract_features(node)

        device = next(self.parameters()).device
        node_features_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(device)

        # Get neural unit for this specific (normalized_type, num_children) combination
        unit = self.get_unit(normalized_type, num_children)

        # Prepare input for neural unit
        if children_outputs:
            # Internal node: concatenate node features with children outputs
            children_concat = torch.cat(children_outputs, dim=1)
            unit_input = torch.cat([node_features_tensor, children_concat], dim=1)
        else:
            # Leaf node: just node features
            unit_input = node_features_tensor

        # Forward pass through neural unit
        output = unit(unit_input)

        # Cache result
        cache[node_id] = output

        return output

    def forward(self, plan: dict) -> torch.Tensor:
        """
        Predict latency for a query plan.

        Args:
            plan: Query plan dictionary (root node)

        Returns:
            Predicted latency (scalar tensor)
        """
        # Process the plan tree
        cache = {}

        device = next(self.parameters()).device
        self.to(device)

        output = self.process_node(plan, cache)

        # Return just the latency (first element)
        latency = output[:, 0]

        return latency

    def get_all_node_predictions(self, plan: dict) -> dict[int, float]:
        """
        Get latency predictions for all nodes in the plan.
        Used for computing the loss function (Equation 7).

        Args:
            plan: Query plan dictionary

        Returns:
            Dictionary mapping node IDs to predicted latencies
        """

        cache = {}
        device = next(self.parameters()).device
        self.to(device)

        self.process_node(plan, cache)
        return cache
