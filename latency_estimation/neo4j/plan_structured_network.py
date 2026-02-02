import torch
import torch.nn as nn
from latency_estimation.common import NnOperator
from latency_estimation.train_config import ModelConfig
from latency_estimation.neo4j.neural_units import Estimation, create_neural_unit
from latency_estimation.neo4j.feature_extractor import FeatureExtractor

class PlanStructuredNetwork(nn.Module):
    """
    Plan-structured neural network for Neo4j query performance estimation.
    Maintains a library of neural units (one per operator type).
    Dynamically assembles neural units to match query plan structure.
    """
    def __init__(self, config: ModelConfig, feature_extractor: FeatureExtractor):
        super().__init__()

        self.config = config
        self.feature_extractor = feature_extractor
        # One neural unit for each operator type
        self.units = nn.ModuleDict()
        self.operators: dict[str, NnOperator] = {}

    @staticmethod
    def from_plans(config: ModelConfig, feature_extractor: FeatureExtractor, plans: list[dict]) -> 'PlanStructuredNetwork':
        """Create neural units by analyzing all operator types and their *number of children* in the training plans."""
        model = PlanStructuredNetwork(config, feature_extractor)

        for operator in OperatorCollector.run(feature_extractor, plans):
            model._add_unit_if_not_exists(operator)

        return model

    @staticmethod
    def from_checkpoint(checkpoint: dict, device: str) -> 'PlanStructuredNetwork':
        """Load model from checkpoint, including neural units."""
        config: ModelConfig = checkpoint['config']
        feature_extractor: FeatureExtractor = checkpoint['feature_extractor']

        model = PlanStructuredNetwork(config, feature_extractor)

        operators: dict[str, NnOperator] = checkpoint['operators']
        for operator in operators.values():
            model._add_unit_if_not_exists(operator)

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

    def forward(self, plan: dict) -> torch.Tensor:
        """
        Implements the model's forward pass. Estimate latency for a query plan.
        Args:
            plan: Query plan dictionary (root node)
        Returns:
            Estimated latency (scalar tensor [1, 1])
        """
        # Process the entire plan tree
        cache = dict[int, Estimation]()
        output = self.__process_plan_node(plan, cache)

        # Root should have estimated latency
        if output.latency is None:
            raise ValueError(f'Root operator {plan.get("operatorType")} did not estimate latency. Make sure root is ProduceResults.')

        return output.latency

    # TODO not used?
    # Note: For Neo4j, we process plans individually since they have different structures. Future optimization could group by structure.
    # def estimate_plan_latency_all_nodes(self, plan: dict) -> dict[int, torch.Tensor]:
    #     """
    #     Get latency estimations for all nodes in the plan.
    #     Used for computing the loss function (Equation 7).
    #     Args:
    #         plan: Query plan dictionary
    #     Returns:
    #         Dictionary mapping node IDs to estimated latencies
    #     """
    #     pass

    def __process_plan_node(self, node: dict, cache: dict[int, Estimation]) -> Estimation:
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
        for child in node.get('children', []):
            child_output = self.__process_plan_node(child, cache)
            # Extract data vector from child
            child_outputs.append(child_output.data)

        node_features = self.feature_extractor.extract_features(node)
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)

        operator = NnOperator(
            type = node.get('operatorType', 'Unknown').replace('@neo4j', ''),
            num_children = len(child_outputs),
            feature_dim = len(node_features),
        )
        unit = self.__get_unit(operator)

        # Concatenate operator features with children data vectors
        if child_outputs:
            # Stack child data vectors: [num_children, data_vec_dim]
            child_data = torch.stack(child_outputs)
            # Flatten to [num_children * data_vec_dim]
            child_data_flat = child_data.flatten()
            # Concatenate with operator features
            unit_input = torch.cat([node_features_tensor, child_data_flat])
        else:
            unit_input = node_features_tensor

        # Add batch dimension: [1, input_dim]
        unit_input = unit_input.unsqueeze(0)

        # Forward pass through neural unit
        output = unit(unit_input)

        # Cache result
        cache[node_id] = output

        return output

    def __get_unit(self, operator: NnOperator) -> nn.Module:
        op_key = operator.key()
        if op_key not in self.units:
            # Potential fallback: could try to find the closest num_children.
            # For now, we'll just error out.
            raise ValueError(f'No neural unit found for operator type/children combination: {op_key}. Available types: {list(self.units.keys())}')

        # Operator consistency check - maybe not needed?
        if op_key not in self.operators:
            raise ValueError(f'Operator not found: {op_key}. Available operators: {list(self.operators.keys())}')
        op_original = self.operators[op_key]
        if op_original.feature_dim != operator.feature_dim:
            raise ValueError(f'Feature dimension mismatch for operator {op_key}: expected {op_original.feature_dim}, got {operator.feature_dim}.')

        return self.units[op_key]

    def _add_unit_if_not_exists(self, operator: NnOperator):
        """Create a neural unit for a specific operator type (unless already exists)."""
        op_key = operator.key()
        if op_key in self.units:
            return

        self.units[op_key] = create_neural_unit(
            op_type = operator.type,
            input_dim = operator.feature_dim,
            num_children = operator.num_children,
            config = self.config,
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

        print(f'\nFound {len(operator_pairs)} unique operator type/children combinations:')
        for op_type, num_children in sorted_pairs:
            print(f'  - {op_type} (children: {num_children})')

        operators = list[NnOperator]()
        for op_type, num_children in sorted_pairs:
            feature_dim = self.feature_extractor.get_feature_dim(op_type)
            operators.append(NnOperator(op_type, num_children, feature_dim))

        return operators

    def __collect_operator_pairs(self, node: dict) -> set[tuple[str, int]]:
        """Recursively collect operator types and their children counts."""
        output = set[tuple[str, int]]()

        # Normalize scan types to 'Scan' for unified neural unit.
        op_type = node.get('operatorType', 'Unknown').replace('@neo4j', '')
        children = node.get('children', [])
        output.add((op_type, len(children)))

        # Recursively scan children.
        for child in children:
            output.update(self.__collect_operator_pairs(child))

        return output
