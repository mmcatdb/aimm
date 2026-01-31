"""
Neural Units for Neo4j Query Performance Prediction.

Key differences from PostgreSQL implementation:
1. No per-operator latency prediction (only root operator predicts latency)
2. All internal units output only data vectors (size 32)
3. Only ProduceResults (root) unit outputs [data_vector (32), latency (1)]
4. Loss is computed only on total query latency: (predicted_total - actual_total)^2
"""
import torch
import torch.nn as nn

class Prediction:
    def __init__(self, data: torch.Tensor, latency: torch.Tensor | None):
        self.data = data
        """Data vector output [batch_size, data_vec_dim]"""
        self.latency = latency
        """Latency prediction [batch_size, 1] (only if is_root=True)"""

class NeuralUnit(nn.Module):
    """
    Base class for operator-level neural units.

    For Neo4j, internal units output only data vectors.
    Only the root unit (ProduceResults) outputs latency prediction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, data_vec_dim: int = 32, num_layers: int = 5, is_root: bool = False):
        """
        Args:
            input_dim: Size of input feature vector
            hidden_dim: Size of hidden layers
            data_vec_dim: Size of output data vector (default 32)
            num_layers: Number of hidden layers
            is_root: Whether this is the root unit (ProduceResults) that predicts latency
        """
        super().__init__()

        self.input_dim = input_dim
        self.data_vec_dim = data_vec_dim
        self.is_root = is_root

        # Build hidden layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer: data vector for all units
        self.data_output = nn.Linear(hidden_dim, data_vec_dim)

        # Latency output: only for root unit
        if is_root:
            self.latency_output = nn.Linear(hidden_dim, 1)
            # Use softplus to ensure positive latency predictions
            self.latency_activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Prediction:
        """
        Forward pass through the neural unit.

        Args:
            x: Input tensor [batch_size, input_dim]
        """
        # Pass through hidden layers
        h = self.hidden_layers(x)

        # Always output data vector
        data_vec = self.data_output(h)

        # Only root unit predicts latency
        latency = None
        if self.is_root:
            latency = self.latency_activation(self.latency_output(h))

        return Prediction(data_vec, latency)

class GenericUnit(NeuralUnit):
    """
    Generic neural unit for Neo4j operators.

    Handles variable number of children by concatenating their data vectors
    with the operator's own features.
    """

    def __init__(self, input_dim: int, num_children: int = 0, data_vec_dim: int = 32, is_root: bool = False, **kwargs):
        """
        Args:
            input_dim: Size of operator's feature vector (without children)
            num_children: Number of child operators
            data_vec_dim: Size of data vector output
            is_root: Whether this is the root operator
            **kwargs: Additional arguments passed to NeuralUnit
        """
        # Total input = operator features + (data_vec_dim * num_children)
        total_input_dim = input_dim + (data_vec_dim * num_children)

        super().__init__(total_input_dim, data_vec_dim=data_vec_dim, is_root=is_root, **kwargs)

        self.num_children = num_children
        self.operator_feature_dim = input_dim

def create_neural_unit(operator_type: str, input_dim: int, num_children: int = 0, data_vec_dim: int = 32, is_root: bool = False, **kwargs) -> NeuralUnit:
    """
    Factory function to create a neural unit for a given operator type.

    For Neo4j, we use a generic unit for all operators.
    The root operator (ProduceResults) is special in that it predicts latency.

    Args:
        operator_type: Neo4j operator type (e.g., 'ProduceResults', 'Filter', etc.)
        input_dim: Size of operator's feature vector
        num_children: Number of child operators
        data_vec_dim: Size of data vector
        is_root: Whether this is the root operator
        **kwargs: Additional arguments

    Returns:
        Neural unit for the operator
    """

    # ProduceResults should always be the root
    is_root_op = (operator_type == 'ProduceResults')

    return GenericUnit(
        input_dim=input_dim,
        num_children=num_children,
        data_vec_dim=data_vec_dim,
        is_root=is_root_op,
        **kwargs
    )
