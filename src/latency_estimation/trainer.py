from abc import ABC, abstractmethod
from collections import defaultdict
import json
import os
from typing import Protocol
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from core.utils import INFO_TEXT, BOLD_TEXT, RESET_TEXT, CLEAR_TEXT_LINE, exit_with_error, print_warning
from .config import TrainerConfig
from .dataset import DatasetItem
from .model import BaseModel, CheckpointId, CheckpointName, create_checkpoint_id

TrainerMetrics = dict[str, float | int]

class IPathProvider(Protocol):
    def model(self, checkpoint_id: CheckpointId) -> str: ...

    def metrics(self, checkpoint_id: CheckpointId) -> str: ...

class BaseTrainer(ABC):
    """
    Base Trainer for plan-structured neural networks.
    Implements optimized training with batching and caching.
    """

    def __init__(self, main_metric: str, train_metrics: list[str], config: TrainerConfig):
        self.__main_metric = main_metric
        """Key of the primary metric used for tracking the best model."""
        self.__train_metrics = train_metrics
        """List of metric keys to print during training."""
        self.__batch_size = config.batch_size
        self.__epoch_period = config.epoch_period
        """Number of epochs between evaluations and potential best model updates."""
        self.__autosave_period = config.autosave_period
        """Number of epochs between automatic checkpoint saves. Should be a multiple of epoch_period."""
        self.__config = config

        self._loss_history = list[float]()
        self.__is_checkpoint_saved = False

    @abstractmethod
    def model(self) -> BaseModel:
        """Return the underlying model being trained."""
        pass

    def to_checkpoint(self) -> dict:
        """Serialization to a file-friendly dictionary."""
        return {
            'epoch': len(self._loss_history),
            'loss_history': self._loss_history,
            'config': self.__config,
        }

    def load_from_checkpoint(self, checkpoint: dict):
        """Load inner state (e.g., optimizer) from the checkpoint dictionary."""
        self._loss_history = checkpoint['loss_history']

    @abstractmethod
    def evaluate(self, dataset: Dataset[DatasetItem]) -> TrainerMetrics:
        """Evaluate the model on the dataset and return metrics."""
        pass

    def train_epochs(self, train_dataset: Dataset[DatasetItem], val_dataset: Dataset[DatasetItem], num_epochs: int, ipp: IPathProvider) -> None:
        # Global access for convenience.
        self._ipp = ipp
        self._create_directories()

        best_metrics: TrainerMetrics = {}
        best_metric = float('inf')

        print()

        for epoch in range(num_epochs):
            epoch_number = epoch + 1
            epoch_prefix = f'Epoch {epoch_number}/{num_epochs}'

            loss = self.train_epoch(train_dataset, epoch_prefix)

            if (epoch_number) % self.__epoch_period == 0:
                print(f'\r{epoch_prefix}{CLEAR_TEXT_LINE}')
                print(f'  Loss: {loss:.4f}')

                # Evaluate on validation set
                metrics = self.evaluate(val_dataset)
                main_metric = metrics[self.__main_metric]

                # Print all metrics
                for metric_key in self.__train_metrics:
                    if metric_key in metrics:
                        print_metric(metrics, metric_key, is_main=(metric_key == self.__main_metric))
                self._save_epoch_metrics(metrics, epoch_number, loss)

                # Track and save best model
                if main_metric < best_metric:
                    best_metric = main_metric
                    best_metrics = metrics
                    self._save_checkpoint(_BEST_CHECKPOINT, metrics)
                    self._save_metrics(_BEST_CHECKPOINT, metrics)
                    print(f'  {INFO_TEXT}✓ New best model!{RESET_TEXT}')

                # Autosave because why not
                if (epoch_number) % self.__autosave_period == 0:
                    self._save_checkpoint(_epoch_checkpoint(epoch_number), metrics)

                print()

        self.__final_evaluation(train_dataset, val_dataset)
        print_metrics(best_metrics, 'Best Validation')

    def train_epoch(self, dataset: Dataset[DatasetItem], epoch_prefix: str) -> float:
        # Identity collate_fn - return list of items as-is
        data_loader = DataLoader(dataset, batch_size=self.__batch_size, shuffle=True, collate_fn=lambda x: x)
        epoch_losses = []

        for index, batch in enumerate(data_loader):
            loss = self._train_batch(batch)
            epoch_losses.append(loss)

            if (index + 1) % 10 == 0:
                # The empty spaces at the end are to overwrite any previous longer output. Like it's not optimal (it's acutally horrendous) but it works.
                print(f'\r{epoch_prefix}, Batch {index + 1}/{len(data_loader)}, Loss: {loss:.6f}{CLEAR_TEXT_LINE}', end='')

        avg_loss = np.mean(epoch_losses).item()
        self._loss_history.append(avg_loss)

        self._after_epoch()

        return avg_loss

    @abstractmethod
    def _train_batch(self, batch: list[DatasetItem]) -> float:
        """Train for one epoch over the dataset."""
        pass

    def _after_epoch(self):
        """Hook for any final steps after epoch completes."""
        pass

    def _create_directories(self):
        """Create necessary directories for saving checkpoints and metrics. Called at the beginning of training."""
        model_id = self.model().model_id
        checkpoint_name = _epoch_checkpoint(0) # Just to get the epoch directory name for the path provider.
        path = self._ipp.model(create_checkpoint_id(model_id, checkpoint_name))
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _save_checkpoint(self, checkpoint_name: CheckpointName, metrics: TrainerMetrics):
        model_id = self.model().model_id
        path = self._ipp.model(create_checkpoint_id(model_id, checkpoint_name))

        checkpoint = _create_checkpoint_dict(self, metrics, checkpoint_name)
        is_first_time = not self.__is_checkpoint_saved

        save_checkpoint(path, checkpoint, is_first_time)

        self.__is_checkpoint_saved = True

        print(f'Model saved to {path}')

    def _save_epoch_metrics(self, metrics: TrainerMetrics, epoch: int, loss: float):
        checkpoint_name = _epoch_checkpoint(epoch)
        # Extended metrics with epoch and loss for convenience during analysis and plotting.
        metrics = metrics | {'epoch': epoch, 'loss': loss}
        self._save_metrics(checkpoint_name, metrics)

    def _save_metrics(self, checkpoint_name: CheckpointName, metrics: TrainerMetrics):
        model_id = self.model().model_id
        path = self._ipp.metrics(create_checkpoint_id(model_id, checkpoint_name))

        try_save_metrics(path, metrics)

    @staticmethod
    def get_epoch_and_loss_from_metrics(metrics: TrainerMetrics) -> tuple[int, float]:
        epoch = metrics.get('epoch')
        loss = metrics.get('loss')
        if epoch is None or loss is None:
            raise ValueError('Metrics must contain "epoch" and "loss" keys.')

        return int(epoch), float(loss)

    def __final_evaluation(self, train_dataset: Dataset[DatasetItem], val_dataset: Dataset[DatasetItem]):
        train_metrics = self.evaluate(train_dataset)
        val_metrics = self.evaluate(val_dataset)

        print('\n' + '=' * 50)
        print('Training completed!')
        print('=' * 50)

        print()
        train_metrics = self.evaluate(train_dataset)
        print_metrics(train_metrics, 'Final Training')

        print()
        val_metrics = self.evaluate(val_dataset)
        print_metrics(val_metrics, 'Final Validation')

    def _group_plans_by_structure(self, batch: list[DatasetItem]) -> dict[int, list[int]]:
        """Group plans in a batch by their structure.

        Returns mapping from structure hash to indexes in batch.
        """
        groups = defaultdict(list)

        for index, item in enumerate(batch):
            groups[item.structure_hash].append(index)

        return groups

EPOCH_DIRECTORY = 'epoch'

def _epoch_checkpoint(epoch: int) -> str:
    return f'{EPOCH_DIRECTORY}/{epoch:04d}'

_BEST_CHECKPOINT = 'best'

def _create_checkpoint_dict(trainer: BaseTrainer, metrics: TrainerMetrics, checkpoint_name: str) -> dict:
    model = trainer.model()
    return {
        'id': create_checkpoint_id(model.model_id, checkpoint_name),
        'model': model.to_checkpoint(),
        'trainer': trainer.to_checkpoint(),
        # Measured metrics (just so that we can see them easily).
        'metrics': metrics,
    }

def save_checkpoint(path: str, dict: dict, is_first_time: bool) -> None:
    if is_first_time and os.path.isfile(path):
        print_warning(f'Overwriting existing checkpoint file at {path}.')

    try:
        torch.save(dict, path)
    except Exception as e:
        # There is no point in continuing if we can't save the checkpoint.
        exit_with_error(f'Could not save checkpoint to {path}.', e)

def load_checkpoint(path: str, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        exit_with_error(f'Model checkpoint not found at {path}. Specify a valid --checkpoint path.')
    except Exception as e:
        exit_with_error(f'Could not load checkpoint from {path}.', e)

#region Metrics

def print_metrics(metrics: TrainerMetrics, type = 'Evaluation'):
    print(f'{type} Metrics:')
    for key in metrics.keys():
        print_metric(metrics, key)
    print()

def print_metric(metrics: TrainerMetrics, key: str, is_main: bool = False):
    value = metrics.get(key)
    if value is None:
        print_warning(f'Metric "{key}" not found in metrics dictionary.')
        return

    string = METRIC_FORMATTERS[key](value)
    if is_main:
        string = BOLD_TEXT + string + RESET_TEXT

    print('  ' + string)

METRIC_FORMATTERS = {
    'epoch':          lambda x: f'Epoch: {x}',
    'loss':           lambda x: f'Loss: {x:.4f}',
    'mae':            lambda x: f'MAE: {x:.2f} ms',
    'mse':            lambda x: f'MSE: {x:.2f}',
    'mre':            lambda x: f'MRE: {x * 100:.2f} %',
    'rmse':           lambda x: f'RMSE: {x:.2f} ms',
    'mean_q':         lambda x: f'Mean R: {x:.2f}',
    'median_q':       lambda x: f'Median R: {x:.2f}',
    'geo_mean_q':     lambda x: f'Geometric Mean R: {x:.2f}',
    'p90_q_error':    lambda x: f'P90 R: {x * 100:.2f} %',
    'p95_q_error':    lambda x: f'P95 R: {x * 100:.2f} %',
    'r_within_1.5':   lambda x: f'R < 1.5: {x * 100:.2f} %',
    'r_within_2.0':   lambda x: f'R < 2.0: {x * 100:.2f} %',
    'r_within_5.0':   lambda x: f'R < 5.0: {x * 100:.2f} %',
    'relative_error': lambda x: f'Relative Error: {x:.2f}',
}

def try_save_metrics(path: str, metrics: TrainerMetrics) -> None:
    try:
        with open(path, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        print_warning(f'Could not save metrics to {path}.', e)

def load_metrics(path: str) -> TrainerMetrics:
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        exit_with_error(f'Metrics file not found at {path}.')
    except Exception as e:
        exit_with_error(f'Could not load metrics from {path}.', e)

#endregion
