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
from latency_estimation.neo4j.model import Model

class Trainer(BaseTrainer):

    def __init__(self, model: Model, config: TrainerConfig):
        super().__init__(
            main_metric='mse',
            train_metrics=['mse', 'rmse', 'mae', 'mean_q', 'median_q'],
            config=config,
        )
        self.__model = model
        self.__optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

        # Loss function: MSE for total query latency
        self.criterion = nn.MSELoss()

    @override
    def model(self) -> Model:
        return self.__model

    @override
    def to_checkpoint(self) -> dict:
        return super().to_checkpoint() | {
            # Weight decay should be saved automatically.
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }

    @override
    def load_from_checkpoint(self, checkpoint: dict):
        super().load_from_checkpoint(checkpoint)
        self.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @override
    def evaluate(self, dataset: Dataset[DatasetItem]) -> TrainerMetrics:
        self.__model.eval()

        predictions = []
        actuals = []

        for item in dataset:
            try:
                predicted = self.__model.evaluate(item.plan)
            except Exception as e:
                print_warning(f'Could not compute model outputs for a query: \n{item.query_id}', e)
                continue

            predictions.append(predicted)
            actuals.append(item.latency)

        # Convert to numpy arrays
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Mean squared error
        mse = np.mean((predictions - actuals) ** 2).item()
        rmse = np.sqrt(mse)
        # Mean absolute error
        mae = np.mean(np.abs(predictions - actuals)).item()
        # R-value: max(pred/actual, actual/pred)
        r_values = np.maximum(
            predictions / (actuals + EPSILON),
            actuals / (predictions + EPSILON),
        )
        # Mean relative error
        mre = np.mean(np.abs(predictions - actuals) / (actuals + EPSILON)).item()

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mre': mre,
            'mean_q': np.mean(r_values).item(),
            'median_q': np.median(r_values).item(),
            'p90_q_error': np.percentile(r_values, 90).item(),
            'p95_q_error': np.percentile(r_values, 95).item(),
        }

    def _train_batch(self, batch: list[DatasetItem]) -> float:
        """Returns the loss for the batch."""
        self.__model.train()
        self.__optimizer.zero_grad()
        loss = self.__compute_loss(batch)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)
        # Update weights
        self.__optimizer.step()

        return loss.item()

    def __compute_loss(self, batch: list[DatasetItem]) -> torch.Tensor:
        """
        Compute MSLE loss for a batch of query plans.
        Loss = (log(estimated + 1) - log(actual + 1))^2
        """
        predictions = []
        actuals = []

        # Group plans by structure for efficiency
        groups = self._group_plans_by_structure(batch)

        for indexes in groups.values():
            # Process plans with same structure together
            for index in indexes:
                item = batch[index]
                actual = item.latency

                try:
                    predicted = self.__model.forward(item.plan)
                except Exception as e:
                    print_warning(f'Could not compute model outputs for a query: \n{item.query_id}', e)
                    continue

                predictions.append(predicted)
                actuals.append(torch.tensor([[actual]], dtype=predicted.dtype, device=predicted.device))

        # Stack predictions and actuals
        predictions = torch.cat(predictions, dim=0)
        actuals = torch.cat(actuals, dim=0)

        # Apply Log transformation before MSE
        predictions_log = torch.log1p(torch.abs(predictions))
        acutals_log = torch.log1p(actuals)

        # Compute MSE on log values -> MSLE
        return self.criterion(predictions_log, acutals_log)
