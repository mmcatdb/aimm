from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):

    @staticmethod
    @abstractmethod
    def get_node_type(node: dict) -> str:
        """Extract the operator type from a plan node."""
        pass

    @staticmethod
    @abstractmethod
    def get_node_children(node: dict) -> list[dict]:
        """Extract child nodes from a plan node."""
        pass

    @abstractmethod
    def get_feature_dim(self, op_type: str) -> int:
        """Get the dimension of the feature vector for a given node type."""
        pass
