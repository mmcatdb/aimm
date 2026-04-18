from typing_extensions import override
import torch
import torch.nn as nn
from core.nn_operator import NnOperator
from core.drivers import DriverType
from ..config import ModelConfig
from ..feature_extractor import PlanNode
from ..model import BaseModel, ModelName, TsneItem, create_model_id

class Model(BaseModel):

    def __init__(self, config: ModelConfig, model_name: ModelName):
        super().__init__(config, create_model_id(DriverType.POSTGRES, model_name))

    @override
    def _create_unit(self, operator: NnOperator) -> nn.Module:
        data_vec_dim = self.config.data_vec_dim
        input_dim = operator.feature_dim + operator.num_children * (1 + data_vec_dim)

        return NeuralUnit(
            input_dim=input_dim,
            output_dim=data_vec_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_num=self.config.hidden_num
        )

    @override
    def forward(self, plan: PlanNode) -> torch.Tensor:
        cache = dict[int, torch.Tensor]()
        output = self.__process_plan_node(plan, cache)
        return output[:, 0]

    def estimate_plan_latency_all_nodes(self, plan: PlanNode) -> dict[int, torch.Tensor]:
        """
        Get latency estimations for all nodes in the plan.
        Used for computing the loss function (Equation 7).
        Args:
            plan: Query plan dictionary
        Returns:
            Dictionary mapping node IDs to estimated latencies
        """
        cache = dict[int, torch.Tensor]()

        self.__process_plan_node(plan, cache)

        return cache

    def get_tsne_data(self, plan: PlanNode) -> list['TsneItem']:
        tsne_items = list[TsneItem]()

        cache = dict[int, torch.Tensor]()

        self.__process_plan_node(plan, cache, tsne_items)

        return tsne_items

    def __process_plan_node(self, node: PlanNode, cache: dict[int, torch.Tensor], tsne_items: list['TsneItem'] | None = None) -> torch.Tensor:
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
        for child in node.children:
            child_outputs.append(self.__process_plan_node(child, cache, tsne_items))

        node_features_tensor = torch.tensor(node.features, dtype=torch.float32, device=self.device).unsqueeze(0)

        operator = NnOperator(
            type=node.type,
            num_children=len(child_outputs),
            feature_dim=len(node.features),
        )
        unit = self._get_unit(operator)

        # Prepare input for neural unit
        if child_outputs:
            children_concat = torch.cat(child_outputs, dim=1)
            unit_input = torch.cat([node_features_tensor, children_concat], dim=1)
        else:
            # Leaf node: just node features
            unit_input = node_features_tensor

        # Forward pass through neural unit
        output: torch.Tensor = unit(unit_input)

        # Cache result
        cache[node_id] = output

        # Add to t-SNE items if provided
        if tsne_items is not None:
            tsne_items.append(TsneItem(
                operator=operator,
                features=node.features,
                extracted=node.latency_checked(),
                estimated=output[:, 0].item(),
            ))

        return output

class NeuralUnit(nn.Module):
    """
    Base class for operator-level neural units.
    Each unit has:
    - Hidden layers for feature learning
    - Latency output (1 value)
    - Data vector output (d values)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_num: int):
        """
        Args:
            input_dim: Dimension of input features (including children outputs)
            hidden_dim: Number of neurons in each hidden layer
            hidden_num: Number of hidden layers
            output_dim: Dimension of output data vector
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.output_dim = output_dim

        # Build hidden layers
        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(hidden_num - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.hidden_layers = nn.Sequential(*layers)

        # Output layers
        self.latency_output = nn.Linear(hidden_dim, 1)
        self.data_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through neural unit.
        Args:
            x: Input tensor [batch_size, input_dim]
        Returns:
            Output tensor [batch_size, 1 + output_dim]
            where output[:, 0] is latency and output[:, 1:] is data vector
        """
        # Pass through hidden layers
        h = self.hidden_layers(x)

        latency = self.latency_output(h)
        data_vec = self.data_output(h)

        return torch.cat([latency, data_vec], dim=1)
