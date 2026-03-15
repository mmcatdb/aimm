import torch
import torch.nn as nn
from latency_estimation.common import NnOperator
from latency_estimation.config import ModelConfig
from latency_estimation.mongo.neural_units import create_neural_unit
from latency_estimation.mongo.feature_extractor import FeatureExtractor
from latency_estimation.exceptions import NeuralUnitNotFoundException

class PlanStructuredNetwork(nn.Module):
    """Plan-structured neural network for MongoDB QPP."""

    def __init__(self, config: ModelConfig, feature_extractor: FeatureExtractor):
        super().__init__()

        self.config = config
        self.feature_extractor = feature_extractor
        self.units = nn.ModuleDict()
        """One neural unit for each operator type"""
        self.operators: dict[str, NnOperator] = {}

    @staticmethod
    def from_plans(config: ModelConfig, feature_extractor: FeatureExtractor, plans: list[dict]) -> 'PlanStructuredNetwork':
        """Create neural units by analyzing all operator types and their *number of children* in the training plans."""
        model = PlanStructuredNetwork(config, feature_extractor)

        for operator in OperatorCollector.run(feature_extractor, plans):
            model.__add_unit_if_not_exists(operator)

        return model

    @staticmethod
    def from_checkpoint(checkpoint: dict, device: str) -> 'PlanStructuredNetwork':
        """Load model from checkpoint, including neural units."""
        config: ModelConfig = checkpoint['config']
        feature_extractor: FeatureExtractor = checkpoint['feature_extractor']

        model = PlanStructuredNetwork(config, feature_extractor)

        operators: dict[str, NnOperator] = checkpoint['operators']
        for operator in operators.values():
            model.__add_unit_if_not_exists(operator)

        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model

    def to_checkpoint(self) -> dict:
        """Serialization to a file-friendly dictionary."""
        return {
            'config': self.config,
            'operators': self.operators,
            'feature_extractor': self.feature_extractor,
            # Learned parameters of the model.
            'model_state_dict': self.state_dict(),
        }

    def print_summary(self):
        """Print a summary of the neural units in the model."""
        print(f'Initialized {len(self.units)} neural units. Total parameters: {self.count_parameters():,}.')

    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, plan_tree: dict, collection_name: str) -> torch.Tensor:
        """
        Predict latency for a query plan.

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
        for child in _get_node_children(node):
            child_output = self.__process_plan_node(child, collection_name, cache)
            # Extract data vector from child
            child_outputs.append(child_output)

        node_features = self.feature_extractor.extract_features(node, collection_name)
        device = next(self.parameters()).device
        node_features_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(device)

        type = node.get('stage', 'UNKNOWN')
        operator = NnOperator(
            type=type,
            num_children=len(child_outputs),
            feature_dim=self.feature_extractor.get_feature_dim(type),
        )
        unit = self.__get_unit(operator)

        if child_outputs:
            child_concat = torch.cat(child_outputs, dim=1)
            unit_input = torch.cat([node_features_tensor, child_concat], dim=1)
        else:
            unit_input = node_features_tensor

        output = unit(unit_input)

        cache[node_id] = output

        return output

    def __get_unit(self, operator: NnOperator) -> nn.Module:
        op_key = operator.key()
        if op_key not in self.units:
            # Potential fallback: could try to find the closest num_children.
            # For now, we'll just error out.
            raise NeuralUnitNotFoundException(operator)

        # Operator consistency check - maybe not needed?
        if op_key not in self.operators:
            raise ValueError(f'Operator not found: {op_key}. Available operators: {list(self.operators.keys())}')
        op_original = self.operators[op_key]
        if op_original.feature_dim != operator.feature_dim:
            raise ValueError(f'Feature dimension mismatch for operator {op_key}: expected {op_original.feature_dim}, got {operator.feature_dim}.')

        return self.units[op_key]

    def __add_unit_if_not_exists(self, operator: NnOperator):
        """Creates a neural unit for a specific operator type (unless already exists)."""
        op_key = operator.key()
        if op_key in self.units:
            return

        self.units[op_key] = create_neural_unit(
            op_type=operator.type,
            input_dim=operator.feature_dim,
            num_children=operator.num_children,
            config=self.config,
        )
        self.operators[op_key] = operator

class OperatorCollector():
    """
    - Assembles neural units into trees.
    - Creates a neural network with structure isomorphic to a query plan.
    - Each operator in the plan is replaced by its corresponding neural unit.
    - The network recursively computes outputs bottom-up through the tree.
    """
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor

    @staticmethod
    def run(feature_extractor: FeatureExtractor, plans: list[dict]) -> list[NnOperator]:
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

        print(f'\nFound {len(sorted_pairs)} operator/children combinations:')
        operators = list[NnOperator]()

        for op_type, num_children in sorted(operator_pairs):
            feature_dim = self.feature_extractor.get_feature_dim(op_type)
            print(f'  - {op_type} (children: {num_children}, feature_dim: {feature_dim})')
            operators.append(NnOperator(op_type, num_children, feature_dim))

        return operators

    def __collect_operator_pairs(self, node: dict) -> set[tuple[str, int]]:
        """Recursively collect operator types and their children counts."""
        output = set[tuple[str, int]]()

        op_type = node.get('stage', 'UNKNOWN')
        children = _get_node_children(node)
        output.add((op_type, len(children)))

        for child in children:
            output.update(self.__collect_operator_pairs(child))

        return output

def _get_node_children(node: dict) -> list[dict]:
    """Get child stages of a plan node (handles both inputStage/inputStages)."""
    children = []
    if 'inputStage' in node:
        children.append(node['inputStage'])
    if 'inputStages' in node:
        children.extend(node['inputStages'])
    return children
