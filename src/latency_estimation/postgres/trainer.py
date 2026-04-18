from typing import cast
from typing_extensions import override
import torch
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from core.utils import EPSILON, print_warning
from latency_estimation.config import TrainerConfig
from latency_estimation.dataset import DatasetItem
from latency_estimation.trainer import BaseTrainer, TrainerMetrics
from latency_estimation.postgres.plan_structured_network import PlanStructuredNetwork

class Trainer(BaseTrainer):

    def __init__(self, model: PlanStructuredNetwork, config: TrainerConfig):
        super().__init__(
            main_metric='mae',
            train_metrics=['mae', 'mre', 'r_within_1.5', 'r_within_2.0'],
            config=config,
        )
        self.__model = model
        self.__optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

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

        # Convert to numpy arrays
        estimations = np.array(estimations)
        actuals = np.array(actuals)

        # Mean absolute error
        mae = np.mean(np.abs(estimations - actuals)).item()
        # R-value: max(pred/actual, actual/pred)
        r_values = np.maximum(
            estimations / (actuals + EPSILON),
            actuals / (estimations + EPSILON),
        )
        # Mean relative error
        mre = np.mean(np.abs(estimations - actuals) / (actuals + EPSILON)).item()

        return {
            'mae': mae,
            'mre': mre,
            'mean_q': np.mean(r_values).item(),
            'median_q': np.median(r_values).item(),
            'r_within_1.5': np.mean(r_values <= 1.5).item(),
            'r_within_2.0': np.mean(r_values <= 2.0).item(),
        }

    def _train_batch(self, batch: list[DatasetItem]) -> float:
        """Returns the loss for the batch."""
        self.__model.train()
        self.__optimizer.zero_grad()

        groups = self._group_plans_by_structure(batch)

        # Compute gradient for each group
        total_loss = 0.0
        total_weight = 0

        for indexes in groups.values():
            group_batch = [batch[i] for i in indexes]
            group_size = len(group_batch)

            # Compute loss for this group
            loss = self.__compute_loss(group_batch)

            # Weighted by group size (for proper gradient estimation)
            weighted_loss = loss * group_size
            total_loss += weighted_loss.item()
            total_weight += group_size

            # Backward pass (accumulates gradients)
            weighted_loss.backward()

        # Normalize gradients by total batch size
        for param in self.__model.parameters():
            if param.grad is not None:
                param.grad /= total_weight

        # Update weights
        self.__optimizer.step()

        # Return average loss
        return total_loss / total_weight if total_weight > 0 else 0.0

    def __compute_loss(self, batch: list[DatasetItem]) -> torch.Tensor:
        """
        Compute L2 loss over all operators in all plans (Equation 7).
        Args:
            batch: List of batch items with plans and ground truth
        Returns:
            Mean squared error loss
        """
        total_squared_error = 0.0
        total_nodes = 0

        for item in batch:
            try:
                all_outputs = self.__model.estimate_plan_latency_all_nodes(item.plan)
            except Exception as e:
                print_warning(f'Could not compute model outputs for a query: \n{item.query_id}', e)
                continue

            # Compute squared errors for all nodes
            for output_tensor in all_outputs.values():
                predicted = output_tensor[0, 0] # Get latency (first element)

                actual = item.plan.latency_checked()
                actual_tensor = torch.tensor(actual, dtype=predicted.dtype, device=predicted.device)

                squared_error = (predicted - actual_tensor) ** 2
                total_squared_error += squared_error
                total_nodes += 1

        # Return RMSE
        if total_nodes > 0:
            mse = cast(torch.Tensor, total_squared_error) / total_nodes
            # return torch.sqrt(mse)
            return torch.sqrt(mse + EPSILON)

        # Return a 0.0 tensor that still requires gradients
        return torch.tensor(0.0, requires_grad=True)
