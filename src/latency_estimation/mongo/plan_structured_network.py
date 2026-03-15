from typing_extensions import override
import torch
import torch.nn as nn
from latency_estimation.common import NnOperator
from latency_estimation.config import ModelConfig
from latency_estimation.plan_structured_network import BasePlanStructuredNetwork, OperatorCollector
from latency_estimation.mongo.feature_extractor import FeatureExtractor

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
        return GenericUnit(
            input_dim=operator.feature_dim,
            num_children=operator.num_children,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            data_vec_dim=self.config.data_vec_dim,
        )

    def forward(self, plan_tree: dict, collection_name: str) -> torch.Tensor:
        """
        Implements the model's forward pass. Estimate latency for a query plan.
        Args:
            plan_tree: the plan tree root node (already extracted from explain)
            collection_name: name of the primary collection
        Returns: scalar predicted latency (ms)
        """
        cache: dict[int, torch.Tensor] = {}
        output = self.__process_plan_node(plan_tree, collection_name, cache)
        return output[:, 0]  # latency

    def __process_plan_node(self, node: dict, collection_name: str, cache: dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Recursively process a plan tree node bottom-up.

        Returns: tensor [1, 1 + data_vec_dim]  (latency + data vector)
        """
        node_id = id(node)
        if node_id in cache:
            return cache[node_id]

        child_outputs = []
        for child in self.feature_extractor.get_node_children(node):
            child_output = self.__process_plan_node(child, collection_name, cache)
            # Extract data vector from child
            child_outputs.append(child_output)

        node_features = self.feature_extractor.extract_features(node, collection_name)
        device = next(self.parameters()).device
        node_features_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(device)

        type = self.feature_extractor.get_node_type(node)
        operator = NnOperator(
            type=type,
            num_children=len(child_outputs),
            feature_dim=self.feature_extractor.get_feature_dim(type),
        )
        unit = self._get_unit(operator)

        if child_outputs:
            child_concat = torch.cat(child_outputs, dim=1)
            unit_input = torch.cat([node_features_tensor, child_concat], dim=1)
        else:
            unit_input = node_features_tensor

        output = unit(unit_input)

        cache[node_id] = output

        return output

class NeuralUnit(nn.Module):
    """
    Base class for operator-level neural units.

    Each unit produces:
    - Hidden layers for feature learning
    - Latency estimate (1 scalar, via Softplus for positivity)
    - Data vector (d values, passed to parent unit)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, data_vec_dim: int):
        """
        Args:
            input_dim: Dimension of input features (including children outputs)
            hidden_dim: Number of neurons in each hidden layer
            num_layers: Number of hidden layers
            data_vec_dim: Dimension of output data vector
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.data_vec_dim = data_vec_dim

        # Build hidden layers
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*layers)

        # Latency head with Softplus to ensure positive predictions
        self.latency_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        # Data vector head
        self.data_head = nn.Linear(hidden_dim, data_vec_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: [batch, 1 + data_vec_dim] where [:,0] is latency, [:,1:] is data vec.
        """
        # Pass through hidden layers
        h = self.hidden_layers(x)

        latency = self.latency_head(h)
        data_vec = self.data_head(h)

        return torch.cat([latency, data_vec], dim=1)

class GenericUnit(NeuralUnit):
    """
    Generic neural unit that handles variable numbers of children.
    Total input = operator features + children * (1 + data_vec_dim).
    """
    def __init__(self, input_dim: int, num_children: int, data_vec_dim: int, **kwargs):
        total_input_dim = input_dim + num_children * (1 + data_vec_dim)

        super().__init__(total_input_dim, data_vec_dim=data_vec_dim, **kwargs)

        self.operator_feature_dim = input_dim
        self.num_children = num_children
