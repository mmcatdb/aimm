from typing_extensions import override
import torch
import torch.nn as nn
from latency_estimation.common import NnOperator
from latency_estimation.config import ModelConfig
from latency_estimation.plan_structured_network import BasePlanStructuredNetwork, OperatorCollector
from latency_estimation.postgres.neural_units import create_neural_unit
from latency_estimation.postgres.feature_extractor import FeatureExtractor

class PlanStructuredNetwork(BasePlanStructuredNetwork[FeatureExtractor]):
    def __init__(self, config: ModelConfig, feature_extractor: FeatureExtractor):
        super().__init__(config, feature_extractor)

    @staticmethod
    def from_plans(config: ModelConfig, feature_extractor: FeatureExtractor, plans: list[dict]) -> 'PlanStructuredNetwork':
        """Create neural units by analyzing all operator types and their *number of children* in the training plans."""
        model = PlanStructuredNetwork(config, feature_extractor)
        model._define_operators(OperatorCollector.run(feature_extractor, plans))
        return model

    @staticmethod
    def from_checkpoint(checkpoint: dict, device: str) -> 'PlanStructuredNetwork':
        """Load model from checkpoint, including neural units."""
        config: ModelConfig = checkpoint['config']
        feature_extractor: FeatureExtractor = checkpoint['feature_extractor']

        model = PlanStructuredNetwork(config, feature_extractor)

        operators: dict[str, NnOperator] = checkpoint['operators']
        model._define_operators(list(operators.values()))

        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model

    @override
    def _create_unit(self, operator: NnOperator) -> nn.Module:
        return create_neural_unit(self.config, operator)

    def forward(self, plan: dict) -> torch.Tensor:
        """
        Implements the model's forward pass. Estimate latency for a query plan.
        Args:
            plan: Query plan dictionary (root node)
        Returns:
            Estimated latency (scalar tensor [1, 1])
        """
        cache: dict[int, torch.Tensor] = {}

        # TODO is this needed?
        device = next(self.parameters()).device
        self.to(device)

        output = self.__process_plan_node(plan, cache)

        # Return just the latency (first element)
        return output[:, 0]

    def estimate_plan_latency_all_nodes(self, plan: dict) -> dict[int, torch.Tensor]:
        """
        Get latency estimations for all nodes in the plan.
        Used for computing the loss function (Equation 7).
        Args:
            plan: Query plan dictionary
        Returns:
            Dictionary mapping node IDs to estimated latencies
        """
        cache: dict[int, torch.Tensor] = {}

        # TODO is this needed?
        device = next(self.parameters()).device
        self.to(device)

        self.__process_plan_node(plan, cache)

        return cache

    def __process_plan_node(self, node: dict, cache: dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Recursively process a query plan node.
        Args:
            node: Query plan node dictionary
            cache: Cache for storing intermediate results (optimization)
        Returns:
            Output tensor [1 + data_vec_dim] containing latency and data vector
        """

        # Check cache (information sharing optimization)
        node_id = id(node)
        if node_id in cache:
            return cache[node_id]

        # Process children recursively
        child_outputs = []
        if 'Plans' in node:
            for child in node['Plans']:
                child_outputs.append(self.__process_plan_node(child, cache))

        node_features = self.feature_extractor.extract_features(node)
        # TODO is this needed?
        device = next(self.parameters()).device
        node_features_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(device)

        operator = NnOperator(
            type=self.feature_extractor.get_node_type(node),
            num_children=len(child_outputs),
            feature_dim=len(node_features),
        )
        unit = self._get_unit(operator)

        # Prepare input for neural unit
        if child_outputs:
            # Internal node: concatenate node features with children outputs
            children_concat = torch.cat(child_outputs, dim=1)
            unit_input = torch.cat([node_features_tensor, children_concat], dim=1)
        else:
            # Leaf node: just node features
            unit_input = node_features_tensor

        # Forward pass through neural unit
        output = unit(unit_input)

        # Cache result
        cache[node_id] = output

        return output
