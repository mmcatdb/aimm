from abc import ABC, abstractmethod
from typing import Generic, Protocol
import numpy as np
from torch.utils.data import Dataset, DataLoader
from common.utils import INFO_TEXT, BOLD_TEXT, RESET_TEXT, CLEAR_TEXT_LINE
from latency_estimation.common import TDatasetItem, print_warning
from latency_estimation.plan_structured_network import BasePlanStructuredNetwork

TrainerMetrics = dict[str, float]

class FileManager(Protocol):
    def epoch_id(self, epoch: int) -> str: ...

    def save_checkpoint(self, suffix: str, trainer: 'BaseTrainer', metrics: TrainerMetrics): ...

    def save_metrics(self, suffix: str, metrics: TrainerMetrics): ...

class BaseTrainer(ABC, Generic[TDatasetItem]):
    """
    Base Trainer for plan-structured neural networks.
    Implements optimized training with batching and caching.
    """

    def __init__(self, main_metric: str, train_metrics: list[str], batch_size: int, epoch_period = 5, autosave_period = 25):
        self.__main_metric = main_metric
        """Name of the primary metric used for tracking the best model."""
        self.__train_metrics = train_metrics
        """List of metric names to print during training."""
        self.__batch_size = batch_size

        self.__epoch_period = epoch_period
        """Number of epochs between evaluations and potential best model updates."""
        self.__autosave_period = autosave_period
        """Number of epochs between automatic checkpoint saves. Should be a multiple of epoch_period."""

        self._loss_history = list[float]()

    @abstractmethod
    def model(self) -> BasePlanStructuredNetwork:
        """Return the underlying model being trained."""
        pass

    @abstractmethod
    def to_checkpoint(self) -> dict:
        """Serialization to a file-friendly dictionary."""
        pass

    def _to_common_checkpoint(self) -> dict:
        return {
            'epoch': len(self._loss_history),
            'loss_history': self._loss_history,
        }

    @staticmethod
    def get_loss_history_from_checkpoint(checkpoint: dict) -> list[float]:
        return checkpoint.get('loss_history', [])

    @abstractmethod
    def evaluate(self, dataset: Dataset[TDatasetItem]) -> TrainerMetrics:
        """Evaluate the model on the dataset and return metrics."""
        pass

    def train_epochs(self, train_dataset: Dataset[TDatasetItem], val_dataset: Dataset[TDatasetItem], num_epochs: int, ctx: FileManager) -> None:
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
                for metric_name in self.__train_metrics:
                    if metric_name in metrics:
                        print_metric(metrics, metric_name, is_main=(metric_name == self.__main_metric))
                ctx.save_metrics(ctx.epoch_id(epoch_number), metrics)

                # Track and save best model
                if main_metric < best_metric:
                    best_metric = main_metric
                    best_metrics = metrics
                    ctx.save_checkpoint('best', self, metrics)
                    ctx.save_metrics('best', metrics)
                    print(f'  {INFO_TEXT}✓ New best model!{RESET_TEXT}')

                # Autosave because why not
                if (epoch_number) % self.__autosave_period == 0:
                    ctx.save_checkpoint(ctx.epoch_id(epoch_number), self, metrics)

                print()

        self.__final_evaluation(train_dataset, val_dataset)
        self.print_metrics(best_metrics, 'Best Validation')

    def train_epoch(self, dataset: Dataset[TDatasetItem], epoch_prefix: str) -> float:
        # Identity collate_fn - return list of items as-is
        dataloader = DataLoader(dataset, batch_size=self.__batch_size, shuffle=True, collate_fn=lambda x: x)
        epoch_losses = []

        for index, batch in enumerate(dataloader):
            loss = self._train_batch(batch)
            epoch_losses.append(loss)

            if (index + 1) % 10 == 0:
                # The empty spaces at the end are to overwrite any previous longer output. Like it's not optimal (it's acutally horrendous) but it works.
                print(f'\r{epoch_prefix}, Batch {index + 1}/{len(dataloader)}, Loss: {loss:.6f}{CLEAR_TEXT_LINE}', end='')

        avg_loss = np.mean(epoch_losses).item()
        self._loss_history.append(avg_loss)

        self._after_epoch()

        return avg_loss

    @abstractmethod
    def _train_batch(self, batch: list[TDatasetItem]) -> float:
        """Train for one epoch over the dataset."""
        pass

    def _after_epoch(self):
        """Hook for any final steps after epoch completes."""
        pass

    def __final_evaluation(self, train_dataset: Dataset[TDatasetItem], val_dataset: Dataset[TDatasetItem]):
        train_metrics = self.evaluate(train_dataset)
        val_metrics = self.evaluate(val_dataset)

        print('\n' + '=' * 50)
        print('Training completed!')
        print('=' * 50)

        print()
        train_metrics = self.evaluate(train_dataset)
        self.print_metrics(train_metrics, 'Final Training')

        print()
        val_metrics = self.evaluate(val_dataset)
        self.print_metrics(val_metrics, 'Final Validation')

    @staticmethod
    def print_metrics(metrics: TrainerMetrics, type = 'Evaluation'):
        print(f'{type} Metrics:')
        for name in metrics.keys():
            print_metric(metrics, name)
        print()

METRIC_FORMATTERS = {
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

def print_metric(metrics: TrainerMetrics, name: str, is_main: bool = False):
    value = metrics.get(name)
    if value is None:
        print_warning(f'Metric "{name}" not found in metrics dictionary.')
        return

    string = METRIC_FORMATTERS[name](value)
    if is_main:
        string = BOLD_TEXT + string + RESET_TEXT

    print('  ' + string)
