import argparse
from core.config import Config
from core.drivers import DriverType

class ModelConfig:
    """Parameters that don't change during training or evaluation."""
    def __init__(self,
        hidden_dim: int = 128,
        hidden_num: int = 5,
        data_vec_dim: int = 32,
    ):
        self.hidden_dim = hidden_dim
        """Number of neurons in each hidden layer of the neural units. In the original paper, 128 seemed to be the optimum."""
        self.hidden_num = hidden_num
        """Number of hidden layers in each neural unit (per operator type). In the original paper, 5 seemd to be the optimum."""
        self.data_vec_dim = data_vec_dim
        """Dimension of the data vector passed from child units to parent units."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--hidden-dim',    type=int, help='Hidden dimension size for neural units.')
        parser.add_argument('--hidden-num',    type=int, help='Number of hidden layers per neural unit.')
        parser.add_argument('--data-vec-dim',  type=int, help='Data vector dimension size.')

    @staticmethod
    def from_arguments(config: Config, args: argparse.Namespace, driver_type: DriverType) -> 'ModelConfig':
        d = default_model(driver_type)
        return ModelConfig(
            hidden_dim=args.hidden_dim or d.hidden_dim,
            hidden_num=args.hidden_num or d.hidden_num,
            data_vec_dim=args.data_vec_dim or d.data_vec_dim,
        )

    def __str__(self) -> str:
        return (
            f'hidden_dim = {self.hidden_dim},\n'
            f'hidden_num = {self.hidden_num},\n'
            f'data_vec_dim = {self.data_vec_dim},\n'
        )

def default_model(driver_type: DriverType) -> ModelConfig:
    if driver_type == DriverType.POSTGRES:
        return ModelConfig()
    elif driver_type == DriverType.MONGO:
        return ModelConfig(
            hidden_num=3,
        )
    elif driver_type == DriverType.NEO4J:
        return ModelConfig()

class TrainerConfig:
    """Parameters for training and data collection."""
    def __init__(self,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        warmup_epochs: int = 10,
        epoch_period: int = 5,
        autosave_period: int = 25
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.epoch_period = epoch_period
        self.autosave_period = autosave_period

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--batch-size',      type=int,   help='Batch size for training.')
        parser.add_argument('--num-epochs',      type=int,   help='Number of training epochs.')
        parser.add_argument('--learning-rate',   type=float, help='Learning rate for optimizer.')
        parser.add_argument('--warmup-epochs',   type=int,   help='Number of warmup epochs with higher learning rate.')
        parser.add_argument('--epoch-period',    type=int,   help='Number of epochs between evaluations and potential best model updates.')
        parser.add_argument('--autosave-period', type=int,   help='Number of epochs between automatic checkpoint saves. Should be a multiple of epoch_period.')

    @staticmethod
    def from_arguments(config: Config, args: argparse.Namespace, driver_type: DriverType) -> 'TrainerConfig':
        d = default_trainer(driver_type)
        return TrainerConfig(
            batch_size=args.batch_size or d.batch_size,
            num_epochs=args.num_epochs or config.num_epochs or d.num_epochs,
            learning_rate=args.learning_rate or d.learning_rate,
            warmup_epochs=args.warmup_epochs or d.warmup_epochs,
            epoch_period=args.epoch_period or d.epoch_period,
            autosave_period=args.autosave_period or d.autosave_period,
        )

    def __str__(self) -> str:
        return (
            f'batch_size = {self.batch_size},\n'
            f'num_epochs = {self.num_epochs},\n'
            f'learning_rate = {self.learning_rate},\n'
            f'warmup_epochs = {self.warmup_epochs},\n'
            f'epoch_period = {self.epoch_period},\n'
            f'autosave_period = {self.autosave_period}\n'
        )

def default_trainer(driver_type: DriverType) -> TrainerConfig:
    if driver_type == DriverType.POSTGRES:
        return TrainerConfig()
    elif driver_type == DriverType.MONGO:
        return TrainerConfig(
            num_epochs=250,
        )
    elif driver_type == DriverType.NEO4J:
        return TrainerConfig()
