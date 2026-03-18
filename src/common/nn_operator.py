class NnOperator:
    """Neural network node operator."""
    def __init__(self, type: str, num_children: int, feature_dim: int):
        self.type = type
        self.num_children = num_children
        self.feature_dim = feature_dim

    def key(self) -> str:
        """Get a unique key for this operator."""
        return self.compute_key(self.type, self.num_children)

    @staticmethod
    def compute_key(type: str, num_children: int) -> str:
        return f'{type}_{num_children}'

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'type': self.type,
            'num_children': self.num_children,
            'feature_dim': self.feature_dim
        }

    @staticmethod
    def from_dict(json: dict) -> 'NnOperator':
        """Create from JSON dictionary."""
        return NnOperator(
            type=json['type'],
            num_children=json['num_children'],
            feature_dim=json['feature_dim']
        )
