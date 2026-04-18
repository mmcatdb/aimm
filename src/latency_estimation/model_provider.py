from typing import TypeVar
from core.config import Config
from core.nn_operator import NnOperator
from core.query import DriverType
from latency_estimation.trainer import BaseTrainer, load_checkpoint, print_metrics
from .config import ModelConfig, TrainerConfig
from .model import BaseModel, ModelName, parse_model_id

TModel = TypeVar('TModel', bound=BaseModel)

class ModelProvider:

    def __init__(self, config: Config, quiet: bool = False):
        self.config = config
        self.quiet = quiet

    def create_model(self, driver_type: DriverType, model_config: ModelConfig, model_name: ModelName, operators: list[NnOperator]) -> BaseModel:
        model = self._crate_model_instance(driver_type, model_config, model_name)
        model.define_operators(operators)
        model.set_device(self.config.device)

        return model

    def _create_model_from_checkpoint(self, checkpoint: dict) -> BaseModel:
        driver_type, model_name = parse_model_id(checkpoint['id'])

        model_config: ModelConfig = checkpoint['config']
        model = self._crate_model_instance(driver_type, model_config, model_name)

        operators: dict[str, NnOperator] = checkpoint['operators']
        model.define_operators(list(operators.values()))

        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.set_device(self.config.device)
        model.eval()

        return model

    def _crate_model_instance(self, driver_type: DriverType, model_config: ModelConfig, model_name: ModelName) -> BaseModel:
        if driver_type == DriverType.POSTGRES:
            from .postgres.model import Model
            return Model(model_config, model_name)
        elif driver_type == DriverType.MONGO:
            from .mongo.model import Model
            return Model(model_config, model_name)
        elif driver_type == DriverType.NEO4J:
            from .neo4j.model import Model
            return Model(model_config, model_name)

    def create_trainer(self, model: BaseModel, trainer_config: TrainerConfig) -> BaseTrainer:
        driver_type, _ = parse_model_id(model.model_id)
        if driver_type == DriverType.POSTGRES:
            from .postgres.model import Model
            from .postgres.trainer import Trainer
            return Trainer(_cast_model(model, Model), trainer_config)
        elif driver_type == DriverType.MONGO:
            from .mongo.model import Model
            from .mongo.trainer import Trainer
            return Trainer(_cast_model(model, Model), trainer_config)
        elif driver_type == DriverType.NEO4J:
            from .neo4j.model import Model
            from .neo4j.trainer import Trainer
            return Trainer(_cast_model(model, Model), trainer_config)

        raise ValueError(f'Unsupported model driver type: {driver_type}')

    def _create_trainer_from_checkpoint(self, model: BaseModel, checkpoint: dict) -> BaseTrainer:
        trainer = self.create_trainer(model, checkpoint['config'])
        trainer.load_from_checkpoint(checkpoint)
        return trainer

    def load_model(self, path: str) -> BaseModel:
        if not self.quiet:
            print('Loading trained model...')

        checkpoint = load_checkpoint(path, self.config.device)
        model = self._create_model_from_checkpoint(checkpoint['model'])

        if not self.quiet:
            print('Model loaded successfully!\n')
            model.print_summary()
            print_metrics(checkpoint['metrics'])

        return model

    def load_trainer(self, path: str) -> BaseTrainer:
        if not self.quiet:
            print('Loading trained model and trainer...')

        checkpoint = load_checkpoint(path, self.config.device)
        model = self._create_model_from_checkpoint(checkpoint['model'])
        trainer = self._create_trainer_from_checkpoint(model, checkpoint['trainer'])

        if not self.quiet:
            print('Model and trainer loaded successfully!\n')
            model.print_summary()
            print_metrics(checkpoint['metrics'])

        return trainer

def _cast_model(model: BaseModel, target: type[TModel]) -> TModel:
    if not isinstance(model, target):
        raise ValueError(f'Expected model of type {target}, got {type(model)}')

    return model
