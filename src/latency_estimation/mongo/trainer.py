from typing_extensions import override
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from core.utils import EPSILON, print_warning
from latency_estimation.config import TrainerConfig
from latency_estimation.dataset import DatasetItem
from latency_estimation.trainer import BaseTrainer, TrainerMetrics
from latency_estimation.mongo.plan_structured_network import PlanStructuredNetwork

class Trainer(BaseTrainer):

    def __init__(self, model: PlanStructuredNetwork, config: TrainerConfig):
        super().__init__(
            main_metric='geo_mean_q', # Use geometric mean Q-error as primary criterion (more robust than MAE)
            train_metrics=['mae', 'median_q', 'r_within_2.0', 'geo_mean_q'],
            config=config,
        )
        self.__model = model
        self.__optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        self.__scheduler = optim.lr_scheduler.CosineAnnealingLR(self.__optimizer, T_max=config.num_epochs, eta_min=config.learning_rate * 0.01)
        self.__warmup_epochs = config.warmup_epochs

    @override
    def model(self) -> PlanStructuredNetwork:
        return self.__model

    @override
    def to_checkpoint(self) -> dict:
        return super().to_checkpoint() | {
            # Momentum should be saved automatically.
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }

    @override
    def load_from_checkpoint(self, checkpoint: dict):
        self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @override
    def evaluate(self, dataset: Dataset[DatasetItem]) -> TrainerMetrics:
        self.__model.eval()

        estimations = []
        actuals = []

        for item in dataset:
            try:
                predicted = self.__model.evaluate(item.plan)
            except Exception as e:
                print_warning(f'Could not compute model outputs for a query: \n{item.query_id}', e)
                continue

            estimations.append(predicted)
            actuals.append(item.latency)

        if not estimations:
            return {'mae': float('inf'), 'relative_error': float('inf')}

        estimations = np.array(estimations)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(estimations - actuals)).item()
        relative_error = np.mean(np.abs(estimations - actuals) / (actuals + EPSILON)).item()

        # R-value: max(pred/actual, actual/pred)
        r_values = np.maximum(
            estimations / (actuals + EPSILON),
            actuals / (estimations + EPSILON),
        )
        # Geometric mean of Q-error
        log_qerror = np.mean(np.log(r_values + EPSILON))

        return {
            'mae': mae,
            'relative_error': relative_error,
            'median_q': np.median(r_values).item(),
            'mean_q': np.mean(r_values).item(),
            'geo_mean_q': np.exp(log_qerror),
            'r_within_1.5': np.mean(r_values <= 1.5).item(),
            'r_within_2.0': np.mean(r_values <= 2.0).item(),
            'r_within_5.0': np.mean(r_values <= 5.0).item(),
        }

    @override
    def _after_epoch(self):
        # Step scheduler after warmup
        epoch = len(self._loss_history)
        if epoch > self.__warmup_epochs:
            self.__scheduler.step()

    def _train_batch(self, batch: list[DatasetItem]) -> float:
        """Returns the loss for the batch."""
        self.__model.train()
        self.__optimizer.zero_grad()
        loss = self.__compute_loss(batch)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=5.0)
        # Update weights
        self.__optimizer.step()

        return loss.item()

    def __compute_loss(self, batch: list[DatasetItem]) -> torch.Tensor:
        """
        Compute log-latency MSE loss over a batch.
        Only uses root-level latency prediction vs actual execution time.

        L = (log(pred+eps) - log(actual+eps))^2
        This optimizes for geometric mean relative error.
        """
        losses = []

        for item in batch:
            actual = item.latency

            try:
                predicted = self.__model.forward(item.plan)
            except Exception as e:
                print_warning(f'Could not compute model outputs for a query: \n{item.query_id}', e)
                continue

            predicted_log = torch.log(predicted + EPSILON)
            actual_log = torch.log(torch.tensor(actual + EPSILON, dtype=predicted_log.dtype, device=predicted_log.device))
            loss = (predicted_log - actual_log) ** 2
            losses.append(loss)

        if losses:
            return torch.stack(losses).mean()

        return torch.tensor(0.0, requires_grad=True)
