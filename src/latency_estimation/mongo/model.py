from typing_extensions import override
import torch
import torch.nn as nn
from core.nn_operator import NnOperator
from core.drivers import DriverType
from ..config import ModelConfig
from ..feature_extractor import PlanNode
from ..model import BaseModel, ModelName, create_model_id

class Model(BaseModel):

    def __init__(self, config: ModelConfig, model_name: ModelName):
        super().__init__(config, create_model_id(DriverType.MONGO, model_name))

    @override
    def _create_unit(self, operator: NnOperator) -> nn.Module:
        data_vec_dim = self.config.data_vec_dim
        total_input_dim = operator.feature_dim + operator.num_children * (1 + data_vec_dim)

        return NeuralUnit(
            input_dim=total_input_dim,
            output_dim=data_vec_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_num=self.config.hidden_num,
        )

    @override
    def forward(self, plan: PlanNode) -> torch.Tensor:
        cache = dict[int, torch.Tensor]()
        output = self.__process_plan_node(plan, cache)
        return output[:, 0]

    def __process_plan_node(self, node: PlanNode, cache: dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Recursively process a plan tree node bottom-up.

        Returns: tensor [1, 1 + data_vec_dim]  (latency + data vector)
        """
        node_id = id(node)
        if node_id in cache:
            return cache[node_id]

        child_outputs = []
        for child in node.children:
            child_outputs.append(self.__process_plan_node(child, cache))

        node_features = node.features
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32, device=self.device).unsqueeze(0)

        unit = self._get_unit(NnOperator(
            type=node.type,
            num_children=len(child_outputs),
            feature_dim=len(node_features),
        ))

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
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num

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

        # Latency head with Softplus to ensure positive predictions
        self.latency_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        # Data vector head
        self.data_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: [batch, 1 + output_dim] where [:,0] is latency, [:,1:] is data vec.
        """
        # Pass through hidden layers
        h = self.hidden_layers(x)

        latency = self.latency_head(h)
        data_vec = self.data_head(h)

        return torch.cat([latency, data_vec], dim=1)
