from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from core.nn_operator import NnOperator
from core.query import DriverType
from .config import ModelConfig
from .feature_extractor import PlanNode

ModelName = str
"""Identifies a model family (within a specific driver).

Each unique training settings (e.g., model architecture, hyperparameters, training procedure, datasets) should correspond to a different model name.
"""

ModelId = str
"""Identifies a model family.

Pattern: {driver_type}/{model_name}.
Example: `postres/sgd_3/epoch/10`, `mongo/lr/best`.
"""

def create_model_id(driver_type: DriverType, model_name: ModelName) -> ModelId:
    return f'{driver_type.value}/{model_name}'

def parse_model_id(id: ModelId) -> tuple[DriverType, ModelName]:
    """Parses a model id into `driver_type`, `model_name`."""
    driver_type_str, model_name = id.split('/', 1)
    return DriverType(driver_type_str), model_name

CheckpointName = str
"""Identifies a specific model checkpoint (within a model family).

Pattern: either a word (e.g., `best`) or in the form of epoch/{epoch_number}.
"""

CheckpointId = str
"""Identifies a specific model checkpoint.

Pattern: {model_id}/{checkpoint_name}.
"""

def create_checkpoint_id(model_id: ModelId, checkpoint_name: CheckpointName) -> CheckpointId:
    return f'{model_id}/{checkpoint_name}'

def parse_checkpoint_id(checkpoint_id: CheckpointId) -> tuple[DriverType, ModelName, CheckpointName]:
    driver_type_str, model_name, checkpoint_name = checkpoint_id.split('/', 2)
    driver_type = DriverType(driver_type_str)
    return driver_type, model_name, checkpoint_name

class BaseModel(nn.Module, ABC):
    """
    Plan-structured neural network for query performance estimation.
    Maintains a library of neural units (one per operator type).
    Dynamically assembles neural units to match query plan structure.
    """
    def __init__(self, config: ModelConfig, model_id: ModelId):
        super().__init__()

        self.config = config
        self.model_id = model_id
        self.units = nn.ModuleDict()
        """One neural unit for each operator type"""
        self.operators = dict[str, NnOperator]()

        self.device = 'cpu'
        """Default device; will be updated when loading checkpoint or moving model."""

    def set_device(self, device: str):
        """Set the device for the model and move it there."""
        self.device = device
        self.to(device)

    def to_checkpoint(self) -> dict:
        """Serialization to a file-friendly dictionary."""
        return {
            'id': self.model_id,
            'config': self.config,
            'operators': self.operators,
            'feature_extractor': self.feature_extractor,
            # Learned parameters of the model.
            'model_state_dict': self.state_dict(),
        }

    def get_units(self) -> list[NnOperator]:
        return list(self.operators.values())

    def print_summary(self):
        """Prints a summary of the model architecture and parameter count."""
        parameters_count = sum(p.numel() for p in self.parameters())
        print(f'Initialized {len(self.units)} neural units. Total parameters: {parameters_count:,}.')
        print(f'Units: {", ".join(self.operators.keys())}')

    def _get_unit(self, operator: NnOperator) -> nn.Module:
        op_key = operator.key()
        if op_key not in self.units:
            # Potential fallback: could try to find the closest num_children.
            # For now, we'll just error out.
            raise NeuralUnitNotFoundException(operator)

        if op_key not in self.operators:
            raise ValueError(f'Operator not found: {op_key}. Available operators: {list(self.operators.keys())}')

        # Operator consistency check - maybe not needed?
        op_original = self.operators[op_key]
        if op_original.feature_dim != operator.feature_dim:
            raise ValueError(f'Feature dimension mismatch for operator {op_key}: expected {op_original.feature_dim}, got {operator.feature_dim}.')

        return self.units[op_key]

    def define_operators(self, operators: list[NnOperator]):
        """Defines neural units for a list of operators."""
        for operator in operators:
            op_key = operator.key()
            if op_key in self.units:
                return

            self.units[op_key] = self._create_unit(operator)
            self.operators[op_key] = operator

    def find_missing_operators(self, plan: PlanNode) -> list[NnOperator]:
        """Returns all operators that are in the plan but not in the model."""
        plan_operators = OperatorCollector.run_quiet([plan])
        missing_operators = []
        for operator in plan_operators:
            if operator.key() not in self.units:
                missing_operators.append(operator)

        return missing_operators

    @abstractmethod
    def _create_unit(self, operator: NnOperator) -> nn.Module:
        pass

    @abstractmethod
    def forward(self, plan: PlanNode) -> torch.Tensor:
        """Implements the model's forward pass. Estimate latency for a query plan.

        Args:
            plan: The plan tree root node (already extracted from explain)
        Returns:
            Estimated latency (scalar tensor [1, 1], in ms)
        """
        pass

    def evaluate(self, plan: PlanNode) -> float:
        """Evaluate the model on a single plan and return the estimated latency."""
        with torch.no_grad():
            return self.forward(plan).item()

    def get_tsne_data(self, plan: PlanNode) -> list['TsneItem']:
        # NICE_TO_HAVE
        raise NotImplementedError('get_tsne_data is not implemented for this model. This method is used for t-SNE visualization of operator embeddings. If you want to use it, please implement it in your model subclass.')

class NeuralUnitNotFoundException(Exception):
    def __init__(self, operator: NnOperator):
        super().__init__(f'Neural unit not found for operator: {operator.key()}.')
        self.operator = operator

@dataclass
class TsneItem:
    operator: NnOperator
    features: np.ndarray
    extracted: float
    estimated: float

class OperatorCollector():
    """
    - Assembles neural units into trees.
    - Creates a neural network with structure isomorphic to a query plan.
    - Each operator in the plan is replaced by its corresponding neural unit.
    - The network recursively computes outputs bottom-up through the tree.
    """

    @staticmethod
    def run(plans: list[PlanNode]) -> list[NnOperator]:
        collector = OperatorCollector()
        output = collector.__collect_unique_operators(plans)

        print(f'\nFound {len(output)} unique operator type/children combinations:')
        for op in output:
            print(f'  - {op.type} (children: {op.num_children}, feature_dim: {op.feature_dim})')

        return output

    @staticmethod
    def run_quiet(plans: list[PlanNode]) -> list[NnOperator]:
        collector = OperatorCollector()
        return collector.__collect_unique_operators(plans)

    def __collect_unique_operators(self, plans: list[PlanNode]) -> list[NnOperator]:
        """Collect all unique operators from the plans."""
        # Pairs of (operator_type, num_children, feature_dim).
        operator_pairs = set[tuple[str, int, int]]()
        for plan in plans:
            operator_pairs.update(self.__collect_operator_pairs(plan))

        # Sort pairs for consistent ordering.
        sorted_pairs = sorted(list(operator_pairs))

        operators = list[NnOperator]()
        type_children_counts = set[tuple[str, int]]()

        for op_type, num_children, feature_dim in sorted_pairs:
            type_children = (op_type, num_children)
            if type_children in type_children_counts:
                raise ValueError(f'Operator type "{op_type}" with {num_children} children appears with multiple feature dimensions. This is not supported. Please check the plans and ensure consistent feature dimensions for the same operator type and children count.')
            type_children_counts.add(type_children)

            operators.append(NnOperator(op_type, num_children, feature_dim))

        return operators

    def __collect_operator_pairs(self, node: PlanNode) -> set[tuple[str, int, int]]:
        """Recursively collect operator types, children counts, and feature dimensions."""
        output = set[tuple[str, int, int]]()
        output.add((node.type, len(node.children), len(node.features)))

        for child in node.children:
            output.update(self.__collect_operator_pairs(child))

        return output
