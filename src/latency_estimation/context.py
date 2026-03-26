import json
import os
from abc import ABC, abstractmethod
import re
from typing import Generic, TypeVar
import torch
from common.config import Config
from common.utils import print_warning, exit_with_error
from common.database import DatabaseInfo
from common.nn_operator import NnOperator
from datasets.databases import find_database
from latency_estimation.plan_structured_network import BasePlanStructuredNetwork
from latency_estimation.trainer import BaseTrainer, TrainerMetrics

TNetwork = TypeVar('TNetwork', bound=BasePlanStructuredNetwork)

class BaseContext(ABC, Generic[TNetwork]):
    def __init__(self, config: Config, quiet: bool):
        self.config = config
        self.quiet = quiet
        self.info: DatabaseInfo

        self.was_checkpoint_saved = False

    def _dataset_path(self, num_queries: int, num_runs: int) -> str:
        return os.path.join(self.config.cache_directory, f'{self.info.id()}_{num_queries}_{num_runs}.pkl')

    def _checkpoint_path(self, suffix: str) -> str:
        return os.path.join(self.config.checkpoints_directory, f'{self.info.id()}_{suffix}.pt')

    def _metrics_path(self, suffix: str) -> str:
        return os.path.join(self.config.metrics_directory, f'{self.info.id()}_{suffix}.json')

    @staticmethod
    def checkpoint_epoch_filename_regex(info: DatabaseInfo) -> re.Pattern:
        return re.compile(f'{info.id()}_e(\\d+).pt')

    @staticmethod
    def metrics_epoch_filename_regex(info: DatabaseInfo) -> re.Pattern:
        return re.compile(f'{info.id()}_e(\\d+).json')

    def epoch_id(self, epoch: int) -> str:
        return f'e{epoch:04d}'

    @staticmethod
    def available_operators_path(config: Config, info: DatabaseInfo) -> str:
        # We don't cache the operators by the dataset because they are shared between different datasets.
        # I.e., the dataset is the one on which the operators were collected, but we might want to use them for a different dataset.
        # NICE_TO_HAVE This is not ideal, we might want to tie the operators to some checkpoint instead. But let's just do it like this for now.
        return os.path.join(config.cache_directory, f'{info.type.value}_operators.json')

    def save_available_operators(self, network: BasePlanStructuredNetwork):
        path = self.available_operators_path(self.config, self.info)

        try:
            with open(path, 'w') as file:
                sorted_operators = sorted(network.get_units(), key=lambda x: x.key())
                operators = [unit.to_dict() for unit in sorted_operators]
                json.dump(operators, file, indent=4)
        except Exception as e:
            print_warning(f'Could not save available operators to {path}.', e)

    # This is static because we need it in some other modules without instantiating the context.
    @staticmethod
    def load_available_operators(config: Config, info: DatabaseInfo) -> list[NnOperator] | None:
        path = BaseContext.available_operators_path(config, info)

        try:
            with open(path, 'r') as file:
                operators_data = json.load(file)
                return [NnOperator.from_dict(data) for data in operators_data]
        except FileNotFoundError:
            print_warning(f'No available operators file found at {path}.')
            return None
        except Exception as e:
            print_warning(f'Could not load available operators from {path}.', e)
            return None

    def database(self):
        return find_database(self.info)

    # These methods don't change the internal state of the context for a good reason - the use case is much more complex than it seems.
    # The trainer has the model. Sometimes we need only the model, sometimes both. But the trainer should not depend on the model - we might want to create a different model.
    # Similarly, the model shouldn't depend on a specific plan or feature extractor.
    # Each component should be responsible for what it saves to the checkpoint. However, in that case, it shouldn't depend on what is saved by other components (e.g., the trainer shouldn't instantiate the model from the checkpoint - it should be given the model).
    # So, we provide "stupid" methods that just save or load the components without any logic.

    def save_checkpoint(self, suffix: str, trainer: BaseTrainer, metrics: TrainerMetrics) -> None:
        path = self._checkpoint_path(suffix)

        save_checkpoint_file(path, {
            'model': trainer.model().to_checkpoint(),
            'trainer': trainer.to_checkpoint(),
            # Measured metrics (just so that we can see them easily).
            'metrics': metrics,
        }, not self.was_checkpoint_saved)

        self.was_checkpoint_saved = True

        if not self.quiet:
            print(f'Model saved to {path}')

    def load_model(self, path: str | None) -> TNetwork:
        if not self.quiet:
            print('Loading trained model...')

        if not path:
            path = self._checkpoint_path('best')

        checkpoint = load_checkpoint_file(path, self.config.device)
        model = self._create_model(checkpoint['model'])

        if not self.quiet:
            print('Model loaded successfully!\n')
            model.print_summary()
            BaseTrainer.print_metrics(checkpoint['metrics'])

        return model

    @abstractmethod
    def _create_model(self, checkpoint_model: dict) -> TNetwork:
        pass

    def load_checkpoint(self, path: str, learning_rate: float, batch_size: int) -> BaseTrainer:
        if not self.quiet:
            print('Loading trained model and trainer...')

        checkpoint = load_checkpoint_file(path, self.config.device)
        model = self._create_model(checkpoint['model'])
        trainer = self._create_trainer(checkpoint['trainer'], model, learning_rate, batch_size)

        if not self.quiet:
            print('Model and trainer loaded successfully!\n')
            model.print_summary()
            BaseTrainer.print_metrics(checkpoint['metrics'])

        return trainer

    @staticmethod
    def get_loss_history_from_checkpoint(checkpoint: dict) -> list[float]:
        return BaseTrainer.get_loss_history_from_checkpoint(checkpoint['trainer'])

    @abstractmethod
    def _create_trainer(self, checkpoint_trainer: dict, model: TNetwork, learning_rate: float, batch_size: int) -> BaseTrainer:
        pass

    def save_metrics(self, suffix: str, metrics: TrainerMetrics) -> None:
        path = self._metrics_path(suffix)
        save_metrics_file(path, metrics)

def save_checkpoint_file(path: str, dict: dict, is_first_time: bool) -> None:
    if is_first_time and os.path.isfile(path):
        print_warning(f'Overwriting existing checkpoint file at {path}.')

    try:
        torch.save(dict, path)
    except Exception as e:
        # There is no point in continuing if we can't save the checkpoint.
        exit_with_error(f'Could not save checkpoint to {path}.', e)

def load_checkpoint_file(path: str, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        exit_with_error(f'Model checkpoint not found at {path}. Specify a valid --checkpoint path.')
    except Exception as e:
        exit_with_error(f'Could not load checkpoint from {path}.', e)

def save_metrics_file(path: str, metrics: TrainerMetrics) -> None:
    try:
        with open(path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        print_warning(f'Could not save metrics to {path}.', e)

def load_metrics_file(path: str) -> TrainerMetrics | None:
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print_warning(f'Metrics file not found at {path}.')
        return None
    except Exception as e:
        print_warning(f'Could not load metrics from {path}.', e)
        return None
