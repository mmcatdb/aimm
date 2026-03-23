import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from common.utils import EPSILON, print_warning
from latency_estimation.mongo.plan_extractor import MongoItem
from latency_estimation.mongo.plan_structured_network import PlanStructuredNetwork
from common.database import MongoQuery

class PlanStructuredTrainer:
    """
    Trainer for MongoDB plan-structured neural networks.

    Loss: log-latency MSE  L = (log(pred+eps) - log(actual+eps))^2
    This optimizes for geometric mean relative error.
    """

    def __init__(self, model: PlanStructuredNetwork, learning_rate: float, batch_size: int, total_epochs: int, warmup_epochs: int):
        self.__model = model
        self.__optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.__loss_history = []
        self.__batch_size = batch_size

        self.__scheduler = optim.lr_scheduler.CosineAnnealingLR(self.__optimizer, T_max=total_epochs, eta_min=learning_rate * 0.01)
        self.__warmup_epochs = warmup_epochs

    @staticmethod
    def load_from_checkpoint(model: PlanStructuredNetwork, checkpoint: dict, learning_rate: float, batch_size: int) -> 'PlanStructuredTrainer':
        trainer = PlanStructuredTrainer(model, learning_rate, batch_size, checkpoint['total_epochs'], checkpoint['warmup_epochs'])
        trainer.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.__loss_history = checkpoint['loss_history']
        return trainer

    def to_checkpoint(self) -> dict:
        """Serialization to a file-friendly dictionary."""
        return {
            'epoch': len(self.__loss_history),
            'loss_history': self.__loss_history,
            # We don't save the learning rate as it can be changed between sessions.
            # Momentum should be saved automatically.
            'optimizer_state_dict': self.__optimizer.state_dict(),
            'total_epochs': self.__scheduler.T_max,
            # TODO This is kinda not ideal as the number of epochs should be changeable between session.
            # However, this feature is not used yet so we can leave it for now.
            'warmup_epochs': self.__warmup_epochs,
        }

    def evaluate(self, dataset: Dataset[MongoItem]) -> dict[str, float]:
        """Evaluate model on a dataset, returning standard QPP metrics."""
        self.__model.eval()

        estimations = []
        actuals = []

        with torch.no_grad():
            for item in dataset:
                query: MongoQuery = item.query

                collection_name = item.query.collection
                plan = item.plan
                actual_ms = item.execution_time

                try:
                    predicted_ms = self.__model(plan, collection_name).item()
                except Exception as e:
                    print_warning(f'Could not compute model outputs for a query: \n{query}', e)
                    continue

                estimations.append(predicted_ms)
                actuals.append(actual_ms)

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

        metrics = {
            'mae': mae,
            'relative_error': relative_error,
            'median_r': np.median(r_values).item(),
            'mean_r': np.mean(r_values).item(),
            'geo_mean_r': np.exp(log_qerror),
            'r_within_1.5': np.mean(r_values <= 1.5).item(),
            'r_within_2.0': np.mean(r_values <= 2.0).item(),
            'r_within_5.0': np.mean(r_values <= 5.0).item(),
        }
        return metrics

    @staticmethod
    def print_metrics(metrics: dict[str, float]):
        print('Evaluation Metrics:')
        print(f'  MAE: {metrics["mae"]:.2f} ms')
        print(f'  Relative Error: {metrics["relative_error"]:.2f}')
        print(f'  Median R: {metrics["median_r"]:.2f}')
        print(f'  Mean R: {metrics["mean_r"]:.2f}')
        print(f'  Geometric Mean R: {metrics["geo_mean_r"]:.2f}')
        print(f'  R<=1.5: {metrics["r_within_1.5"] * 100:.1f}%')
        print(f'  R<=2.0: {metrics["r_within_2.0"] * 100:.1f}%')
        print(f'  R<=5.0: {metrics["r_within_5.0"] * 100:.1f}%')
        print('')

    def train_epoch(self, dataset: Dataset[MongoItem], batch_size: int | None = None, shuffle: bool = True) -> float:
        batch_size = batch_size if batch_size is not None else self.__batch_size
        # Identity collate_fn - return list of items as-is
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)

        epoch_losses = []

        for index, batch in enumerate(dataloader):
            loss = self.__train_batch(batch)
            epoch_losses.append(loss)

            if (index + 1) % 10 == 0:
                print(f'  Batch {index + 1}/{len(dataloader)}, Loss: {loss:.6f}')

        avg_loss = np.mean(epoch_losses).item() if epoch_losses else 0.0
        self.__loss_history.append(avg_loss)

        # Step scheduler after warmup
        epoch = len(self.__loss_history)
        if epoch > self.__warmup_epochs:
            self.__scheduler.step()

        return avg_loss

    def __train_batch(self, batch: list[MongoItem]) -> float:
        self.__model.train()
        self.__optimizer.zero_grad()
        loss = self.__compute_loss(batch)
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=5.0)
        self.__optimizer.step()

        return loss.item()

    def __compute_loss(self, batch: list[MongoItem]) -> torch.Tensor:
        """
        Compute log-latency MSE loss over a batch.
        Only uses root-level latency prediction vs actual execution time.
        """
        losses = []

        for item in batch:
            query: MongoQuery = item.query
            collection_name = query.collection
            plan = item.plan
            actual_ms = item.execution_time

            try:
                predicted_ms = self.__model(plan, collection_name)
            except Exception as e:
                print_warning(f'Could not compute model outputs for a query: \n{query}', e)
                continue

            log_pred = torch.log(predicted_ms + EPSILON)
            log_actual = torch.log(torch.tensor(actual_ms + EPSILON, dtype=log_pred.dtype, device=log_pred.device))
            loss = (log_pred - log_actual) ** 2
            losses.append(loss)

        if losses:
            return torch.stack(losses).mean()

        return torch.tensor(0.0, requires_grad=True)
