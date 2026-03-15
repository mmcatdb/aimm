from typing_extensions import override
import torch
import torch.nn as nn
from latency_estimation.common import NnOperator
from latency_estimation.config import ModelConfig
from latency_estimation.plan_structured_network import BasePlanStructuredNetwork, OperatorCollector
from latency_estimation.neo4j.neural_units import Estimation, create_neural_unit
from latency_estimation.neo4j.feature_extractor import FeatureExtractor

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
        # Process the entire plan tree
        cache: dict[int, Estimation] = {}
        output = self.__process_plan_node(plan, cache)

        # Root should have estimated latency
        if output.latency is None:
            raise ValueError(f'Root operator {plan.get("operatorType")} did not estimate latency. Make sure root is ProduceResults.')

        return output.latency

    # TODO not used?
    # Note: For Neo4j, we process plans individually since they have different structures. Future optimization could group by structure.
    # def estimate_plan_latency_all_nodes(self, plan: dict) -> dict[int, torch.Tensor]:
    #     """
    #     Get latency estimations for all nodes in the plan.
    #     Used for computing the loss function (Equation 7).
    #     Args:
    #         plan: Query plan dictionary
    #     Returns:
    #         Dictionary mapping node IDs to estimated latencies
    #     """
    #     pass

    def __process_plan_node(self, node: dict, cache: dict[int, Estimation]) -> Estimation:
        """
        Recursively process a query plan node.
        Args:
            node: Neo4j query plan node
            cache: Optional cache for node outputs (for batch processing)
        """
        # Generate a unique ID for this node (for caching)
        node_id = id(node)
        if node_id in cache:
            return cache[node_id]

        # Process children recursively
        child_outputs = []
        for child in self.feature_extractor.get_node_children(node):
            child_output = self.__process_plan_node(child, cache)
            # Extract data vector from child
            child_outputs.append(child_output.data)

        node_features = self.feature_extractor.extract_features(node)
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

        operator = NnOperator(
            type=self.feature_extractor.get_node_type(node),
            num_children=len(child_outputs),
            feature_dim=len(node_features),
        )
        unit = self._get_unit(operator)

        # Concatenate operator features with children data vectors
        if child_outputs:
            # Stack child data vectors: [num_children, data_vec_dim]
            child_data = torch.stack(child_outputs)
            # Flatten to [num_children * data_vec_dim]
            child_data_flat = child_data.flatten()
            # Concatenate with operator features
            unit_input = torch.cat([node_features_tensor, child_data_flat])
        else:
            unit_input = node_features_tensor

        # Add batch dimension: [1, input_dim]
        unit_input = unit_input.unsqueeze(0)

        # Forward pass through neural unit
        output = unit(unit_input)

        # Cache result
        cache[node_id] = output

        return output
