from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import torch.nn as nn
from common.nn_operator import NnOperator
from latency_estimation.config import ModelConfig
from latency_estimation.feature_extractor import BaseFeatureExtractor
from latency_estimation.exceptions import NeuralUnitNotFoundException

TExtractor = TypeVar('TExtractor', bound=BaseFeatureExtractor)

class BasePlanStructuredNetwork(nn.Module, ABC, Generic[TExtractor]):
    """
    Plan-structured neural network for query performance estimation.
    Maintains a library of neural units (one per operator type).
    Dynamically assembles neural units to match query plan structure.
    """
    def __init__(self, config: ModelConfig, feature_extractor: TExtractor):
        super().__init__()

        self.config = config
        self.feature_extractor = feature_extractor
        self.units = nn.ModuleDict()
        """One neural unit for each operator type"""
        self.operators: dict[str, NnOperator] = {}

        self.device = 'cpu'
        """Default device; will be updated when loading checkpoint or moving model."""

    def set_device(self, device: str):
        """Set the device for the model and move it there."""
        self.device = device
        self.to(device)

    def to_checkpoint(self) -> dict:
        """Serialization to a file-friendly dictionary."""
        return {
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

    def _define_operators(self, operators: list[NnOperator]):
        """Defines neural units for a list of operators."""
        for operator in operators:
            op_key = operator.key()
            if op_key in self.units:
                return

            self.units[op_key] = self._create_unit(operator)
            self.operators[op_key] = operator

    def find_missing_operators(self, plan: dict) -> list[NnOperator]:
        """Returns all operators that are in the plan but not in the model."""
        plan_operators = OperatorCollector.run_quiet(self.feature_extractor, [plan])
        missing_operators = []
        for operator in plan_operators:
            if operator.key() not in self.units:
                missing_operators.append(operator)

        return missing_operators

    @abstractmethod
    def _create_unit(self, operator: NnOperator) -> nn.Module:
        pass

class OperatorCollector():
    """
    - Assembles neural units into trees.
    - Creates a neural network with structure isomorphic to a query plan.
    - Each operator in the plan is replaced by its corresponding neural unit.
    - The network recursively computes outputs bottom-up through the tree.
    """
    def __init__(self, feature_extractor: BaseFeatureExtractor):
        self.feature_extractor = feature_extractor

    @staticmethod
    def run(feature_extractor: BaseFeatureExtractor, plans: list[dict]) -> list[NnOperator]:
        collector = OperatorCollector(feature_extractor)
        output = collector.__collect_unique_operators(plans)

        print(f'\nFound {len(output)} unique operator type/children combinations:')
        for op in output:
            print(f'  - {op.type} (children: {op.num_children}, feature_dim: {op.feature_dim})')

        return output

    @staticmethod
    def run_quiet(feature_extractor: BaseFeatureExtractor, plans: list[dict]) -> list[NnOperator]:
        collector = OperatorCollector(feature_extractor)
        return collector.__collect_unique_operators(plans)

    def __collect_unique_operators(self, plans: list[dict]) -> list[NnOperator]:
        """Collect all unique operators from the plans."""
        # Pairs of (operator_type, num_children).
        operator_pairs = set[tuple[str, int]]()
        for plan in plans:
            operator_pairs.update(self.__collect_operator_pairs(plan))

        # Sort pairs for consistent ordering.
        sorted_pairs = sorted(list(operator_pairs))

        operators = list[NnOperator]()

        for op_type, num_children in sorted_pairs:
            feature_dim = self.feature_extractor.get_feature_dim(op_type)
            operators.append(NnOperator(op_type, num_children, feature_dim))

        return operators

    def __collect_operator_pairs(self, node: dict) -> set[tuple[str, int]]:
        """Recursively collect operator types and their children counts."""
        output = set[tuple[str, int]]()

        op_type = self.feature_extractor.get_node_type(node)
        children = self.feature_extractor.get_node_children(node)
        output.add((op_type, len(children)))

        for child in children:
            output.update(self.__collect_operator_pairs(child))

        return output
