import torch
import torch.nn as nn
from latency_estimation.config import ModelConfig
from latency_estimation.common import NnOperator

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

def create_neural_unit(config: ModelConfig, operator: NnOperator) -> NeuralUnit:
    """Creates a neural unit for a given operator type."""
    return GenericUnit(
        input_dim=operator.feature_dim,
        num_children=operator.num_children,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        data_vec_dim=config.data_vec_dim,
    )
