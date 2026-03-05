import torch
import torch.nn as nn


class NeuralUnit(nn.Module):
    """
    Base class for operator-level neural units.

    Each unit produces:
    - Latency estimate (1 scalar, via Softplus for positivity)
    - Data vector (d values, passed to parent unit)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, data_vec_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.data_vec_dim = data_vec_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)

        # Latency head with Softplus to ensure positive predictions
        self.latency_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        # Data vector head
        self.data_head = nn.Linear(hidden_dim, data_vec_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: [batch, 1 + data_vec_dim] where [:,0] is latency, [:,1:] is data vec.
        """
        h = self.hidden_layers(x)
        latency = self.latency_head(h)
        data_vec = self.data_head(h)
        return torch.cat([latency, data_vec], dim=1)


class GenericUnit(NeuralUnit):
    """
    Generic neural unit that handles variable numbers of children.
    Total input = operator features + children * (1 + data_vec_dim).
    """

    def __init__(self, input_dim: int, num_children: int = 0,
                 data_vec_dim: int = 32, **kwargs):
        total_input_dim = input_dim + num_children * (1 + data_vec_dim)
        super().__init__(total_input_dim, data_vec_dim=data_vec_dim, **kwargs)
        self.operator_feature_dim = input_dim
        self.num_children = num_children
