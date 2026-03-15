import torch
import torch.nn as nn
from latency_estimation.config import ModelConfig

class NeuralUnit(nn.Module):
    """
    Base class for operator-level neural units.

    Each unit has:
    - Hidden layers for feature learning
    - Latency output (1 value)
    - Data vector output (d values)
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

        # Output layers
        self.latency_output = nn.Linear(hidden_dim, 1)
        self.data_output = nn.Linear(hidden_dim, data_vec_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through neural unit.
        Args:
            x: Input tensor [batch_size, input_dim]
        Returns:
            Output tensor [batch_size, 1 + data_vec_dim]
            where output[:, 0] is latency and output[:, 1:] is data vector
        """
        # Pass through hidden layers
        h = self.hidden_layers(x)

        latency = self.latency_output(h)
        data_vec = self.data_output(h)

        return torch.cat([latency, data_vec], dim=1)

class GenericUnit(NeuralUnit):
    """
    Generic neural unit for all operator types.
    Handles variable number of children by calculating the
    total input dimension correctly.
    """

    def __init__(self, input_dim: int, num_children: int, data_vec_dim: int, **kwargs):
        """
        Args:
            input_dim: Feature dimension of *this operator only*
            num_children: Number of children this operator has
            data_vec_dim: Dimension of data output vectors
            **kwargs: Passed to NeuralUnit (hidden_dim, num_layers, etc.)
        """
        total_input_dim = input_dim + num_children * (1 + data_vec_dim)

        super().__init__(total_input_dim, data_vec_dim=data_vec_dim, **kwargs)

def create_neural_unit(op_type: str, input_dim: int, num_children: int, config: ModelConfig):
    """Creates a neural unit for a given operator type."""
    return GenericUnit(
        input_dim=input_dim,
        num_children=num_children,
        data_vec_dim=config.data_vec_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )
