from argparse import ArgumentParser, Namespace

class ModelConfig:
    """Parameters that don't change during training or evaluation."""
    def __init__(self,
        hidden_dim: int = 128,
        num_layers: int = 5,
        data_vec_dim: int = 32,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.data_vec_dim = data_vec_dim

class TrainConfig:
    """Parameters for training and data collection."""
    def __init__(self,
        num_queries: int,
        train_split: float = 0.8,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        model: ModelConfig = ModelConfig(),
    ):
        self.num_queries = num_queries
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.model = model

    @staticmethod
    def postgres() -> 'TrainConfig':
        return TrainConfig(
            num_queries = 500,
        )

    @staticmethod
    def neo4j() -> 'TrainConfig':
        return TrainConfig(
            num_queries = 250,
        )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('--num-queries', type=int, default=self.num_queries, help='Number of queries to collect')
        parser.add_argument('--train-split', type=float, default=self.train_split, help='Fraction of data for training')
        parser.add_argument('--batch-size', type=int, default=self.batch_size, help='Batch size for training')
        parser.add_argument('--num-epochs', type=int, default=self.num_epochs, help='Number of training epochs')
        parser.add_argument('--learning-rate', type=float, default=self.learning_rate, help='Learning rate for optimizer')
        parser.add_argument('--hidden-dim', type=int, default=self.model.hidden_dim, help='Hidden dimension size')
        parser.add_argument('--num-layers', type=int, default=self.model.num_layers, help='Number of hidden layers per neural unit')
        parser.add_argument('--data-vec-dim', type=int, default=self.model.data_vec_dim, help='Data vector dimension size')
        parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for inference')

    @staticmethod
    def from_arguments(args: Namespace) -> 'TrainConfig':
        return TrainConfig(
            num_queries = args.num_queries,
            train_split = args.train_split,
            batch_size = args.batch_size,
            num_epochs = args.num_epochs,
            learning_rate = args.learning_rate,
            model = ModelConfig(
                hidden_dim = args.hidden_dim,
                num_layers = args.num_layers,
                data_vec_dim = args.data_vec_dim,
            ),
        )

    def __str__(self) -> str:
        return (
            f'num_queries = {self.num_queries},\n'
            f'train_split = {self.train_split},\n'
            f'batch_size = {self.batch_size},\n'
            f'num_epochs = {self.num_epochs},\n'
            f'learning_rate = {self.learning_rate},\n'
            f'hidden_dim = {self.model.hidden_dim},\n'
            f'num_layers = {self.model.num_layers},\n'
            f'data_vec_dim = {self.model.data_vec_dim},\n'
        )
