from abc import ABC, abstractmethod
import os
import pickle
from typing_extensions import Self
import numpy as np

class PlanNode:
    """Extracted features from the query plan tree optimized for ML training. Should be in the shape of the final neural network."""

    def __init__(self, type: str, features: np.ndarray, latency: float | None):
        # TODO make the latency non-nullable.
        self.type = type
        self.features = features
        self.latency = latency
        self.children: list[Self] = []

        self.rows: int # TODO estimate this?
        # postgres: plan.get("Plan Rows", "N/A")
        # TODO postgres total cost: plan.get("Total Cost", "N/A")

    def add_child(self, child: Self):
        self.children.append(child)

    def latency_checked(self) -> float:
        if self.latency is None:
            raise ValueError(f'Latency is not available for node of type "{self.type}".')

        return self.latency

class BaseFeatureExtractor(ABC):

    @abstractmethod
    def extend_vocabularies(self, plans: list[dict], global_stats: dict):
        """Adds data from plans and the global stats to internal vocabularies.

        Must be called before extracting features. Should be called separately on each scaled schema data."""
        pass

    def set_global_stats(self, global_stats: dict):
        """Sets the global statistics for feature extraction. Should be called before extracting features for each specific schema-scale combination."""
        self._global_stats = global_stats

    @abstractmethod
    def extract_plan(self, plan: dict) -> PlanNode:
        """Extract features from the query plan and return the root of the feature tree."""
        pass

    def compute_plan_structure_hash(self, plan: PlanNode) -> int:
        """Computes a hash representing the structure of a query plan.

        Plans with identical structure can be batched together.
        """
        return hash(self._node_structure_string(plan))

    def _node_structure_string(self, node: PlanNode) -> str:
        # Sort for consistency - the order should not matter for the structure hash.
        child_strings = sorted([self._node_structure_string(child) for child in node.children])
        return f'{node.type}_{len(node.features)}({",".join(child_strings)})'

    def _one_hot(self, value: str, vocabulary: set) -> np.ndarray:
        """Encodes a categorical value as a one-hot vector. Returns a binary vector where one (or none) position is 1."""
        return self._multi_hot([value], vocabulary)

    def _multi_hot(self, values: list[str], vocabulary: set) -> np.ndarray:
        """Encodes a list of categorical values as a multi-hot vector. Returns a binary vector where multiple positions can be 1."""
        vocab_list = sorted(vocabulary)
        vector = np.zeros(len(vocab_list), dtype=np.float32)

        for value in values:
            if value in vocab_list:
                index = vocab_list.index(value)
                vector[index] = 1.0

        return vector

    @staticmethod
    @abstractmethod
    def get_node_type(node: dict) -> str:
        """Extracts the operator type from a plan node."""
        pass

    @staticmethod
    @abstractmethod
    def get_node_children(node: dict) -> list[dict]:
        """Extracts child nodes from a plan node."""
        pass

def save_feature_extractor(path: str, feature_extractor: BaseFeatureExtractor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(feature_extractor, file)

def load_feature_extractor(path: str) -> BaseFeatureExtractor:
    with open(path, 'rb') as file:
        return pickle.load(file)
