from typing_extensions import override
import torch
import torch.nn as nn
from core.nn_operator import NnOperator
from core.drivers import DriverType
from ..config import ModelConfig
from ..feature_extractor import PlanNode
from ..plan_structured_network import BasePlanStructuredNetwork, ModelName, create_model_id

class PlanStructuredNetwork(BasePlanStructuredNetwork):

    def __init__(self, config: ModelConfig, model_name: ModelName):
        super().__init__(config, create_model_id(DriverType.NEO4J, model_name))

    @override
    def _create_unit(self, operator: NnOperator) -> nn.Module:
        # The root operator (ProduceResults) is special in that it estimates latency.
        # ProduceResults should always be the root.
        is_root_op = (operator.type == 'ProduceResults')
        data_vec_dim = self.config.data_vec_dim
        input_dim = operator.feature_dim + (data_vec_dim * operator.num_children)

        return NeuralUnit(
            input_dim=input_dim,
            output_dim=self.config.data_vec_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_num=self.config.hidden_num,
            is_root=is_root_op,
        )

    @override
    def forward(self, plan: PlanNode) -> torch.Tensor:
        cache = dict[int, Estimation]()
        output = self.__process_plan_node(plan, cache)

        # Root should have estimated latency
        if output.latency is None:
            raise ValueError(f'Root operator {plan.type} did not estimate latency. Make sure root is ProduceResults.')

        return output.latency

    # TODO not used?
    # Note: For Neo4j, we process plans individually since they have different structures. Future optimization could group by structure.
    # def estimate_plan_latency_all_nodes(self, plan: PlanNode) -> dict[int, torch.Tensor]:
    #     """
    #     Get latency estimations for all nodes in the plan.
    #     Used for computing the loss function (Equation 7).
    #     Args:
    #         plan: Query plan dictionary
    #     Returns:
    #         Dictionary mapping node IDs to estimated latencies
    #     """
    #     pass

    def __process_plan_node(self, node: PlanNode, cache: dict[int, 'Estimation']) -> 'Estimation':
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
        for child in node.children:
            child_output = self.__process_plan_node(child, cache)
            # Extract data vector from child
            child_outputs.append(child_output.data)

        # Add batch dimension: [1, input_dim]
        node_features_tensor = torch.tensor(node.features, dtype=torch.float32, device=self.device).unsqueeze(0)

        unit = self._get_unit(NnOperator(
            type=node.type,
            num_children=len(child_outputs),
            feature_dim=len(node.features),
        ))

        if child_outputs:
            # Concatenate child data vectors
            child_concat = torch.cat(child_outputs, dim=1)
            # Concatenate with operator features
            unit_input = torch.cat([node_features_tensor, child_concat], dim=1)
        else:
            unit_input = node_features_tensor

        # Forward pass through neural unit
        output = unit(unit_input)

        # Cache result
        cache[node_id] = output

        return output

class Estimation:
    def __init__(self, data: torch.Tensor, latency: torch.Tensor | None):
        self.data = data
        """Data vector output [batch_size, data_vec_dim]"""
        self.latency = latency
        """Latency estimation [batch_size, 1] (only if is_root=True)"""

class NeuralUnit(nn.Module):
    """
    Base class for operator-level neural units.

    For Neo4j, internal units output only data vectors.
    Only the root unit (ProduceResults) outputs latency estimation.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_num: int, is_root: bool):
        """
        Args:
            input_dim: Size of input feature vector
            hidden_dim: Size of hidden layers
            output_dim: Size of output data vector (default 32)
            hidden_num: Number of hidden layers
            is_root: Whether this is the root unit (ProduceResults) that estimates latency
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.output_dim = output_dim
        self.is_root = is_root

        # Build hidden layers
        layers = []
        current_dim = input_dim

        for _ in range(hidden_num):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer: data vector for all units
        self.data_output = nn.Linear(hidden_dim, output_dim)

        # Latency output: only for root unit
        if is_root:
            self.latency_output = nn.Linear(hidden_dim, 1)
            # Use softplus to ensure positive latency estimations
            self.latency_activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Estimation:
        """
        Forward pass through the neural unit.
        Args:
            x: Input tensor [batch_size, input_dim]
        """
        # Pass through hidden layers
        h = self.hidden_layers(x)

        # Always output data vector
        data_vec = self.data_output(h)

        # Only root unit estimates latency
        latency = None
        if self.is_root:
            latency = self.latency_activation(self.latency_output(h))

        return Estimation(data_vec, latency)
