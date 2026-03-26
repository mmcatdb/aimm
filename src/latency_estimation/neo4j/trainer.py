from typing_extensions import override
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
from common.utils import EPSILON, print_warning
from latency_estimation.trainer import BaseTrainer, TrainerMetrics
from latency_estimation.neo4j.plan_extractor import Neo4jItem
from latency_estimation.neo4j.feature_extractor import FeatureExtractor
from latency_estimation.neo4j.plan_structured_network import PlanStructuredNetwork

class Trainer(BaseTrainer[Neo4jItem]):

    def __init__(self, model: PlanStructuredNetwork, learning_rate: float, batch_size: int):
        super().__init__(
            main_metric='mse',
            train_metrics=['mse', 'rmse', 'mae', 'mean_q', 'median_q'],
            batch_size=batch_size,
        )
        self.__model = model
        self.__optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Loss function: MSE for total query latency
        self.criterion = nn.MSELoss()

    @staticmethod
    def load_from_checkpoint(model: PlanStructuredNetwork, checkpoint: dict, learning_rate: float, batch_size: int) -> 'Trainer':
        trainer = Trainer(model, learning_rate, batch_size)
        trainer.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer._loss_history = checkpoint['loss_history']
        return trainer

    @override
    def model(self) -> PlanStructuredNetwork:
        return self.__model

    @override
    def to_checkpoint(self) -> dict:
        return self._to_common_checkpoint() | {
            # We don't save the learning rate as it can be changed between sessions.
            # Weight decay should be saved automatically.
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }

    def evaluate(self, dataset: Dataset[Neo4jItem]) -> TrainerMetrics:
        """
        Evaluate model on a dataset.
        Returns:
            Dictionary with evaluation metrics
        """
        self.__model.eval()

        predictions = []
        actuals = []

        with torch.no_grad():
            for item in dataset:
                try:
                    predicted_ms = self.__model(item.plan).item()
                except Exception as e:
                    print_warning(f'Could not compute model outputs for a query: \n{item.query}', e)
                    continue

                predictions.append(predicted_ms)
                actuals.append(item.execution_time)

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
            actuals / (predictions + EPSILON)
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

    def _train_batch(self, batch: list[Neo4jItem]) -> float:
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

    def __compute_loss(self, batch: list[Neo4jItem]) -> torch.Tensor:
        """
        Compute MSLE loss for a batch of query plans.
        Loss = (log(estimated + 1) - log(actual + 1))²
        """
        predictions = []
        actuals = []

        # Group plans by structure for efficiency
        structure_groups = group_plans_by_structure(batch)

        for structure_hash, indexes in structure_groups.items():
            # Process plans with same structure together
            for index in indexes:
                item = batch[index]
                actual = item.execution_time

                try:
                    # TODO tensors vs floats? Why no .item()?
                    predicted = self.__model(item.plan)
                except Exception as e:
                    print_warning(f'Could not compute model outputs for a query: \n{item.query}', e)
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

def group_plans_by_structure(batch: list[Neo4jItem]) -> dict[str, list[int]]:
    """
    Group plans in a batch by their structure.
    Plans with identical structure can share computation.
    Args:
        batch: List of batch items (dicts with 'plan' key)
    Returns:
        Mapping from structure hash to indexes in batch
    """
    groups = defaultdict(list)

    for index, item in enumerate(batch):
        structure = compute_plan_structure_hash(item.plan)
        groups[structure].append(index)

    return groups

def compute_plan_structure_hash(plan: dict) -> str:
    """
    Compute a hash representing the structure of a query plan.
    Plans with identical structure can be batched together.
    Args:
        plan: Neo4j query plan (root node)
    Returns:
        Hash string representing the plan structure
    """
    def structure_sig(node):
        op_type = FeatureExtractor.get_node_type(node)
        children = FeatureExtractor.get_node_children(node)

        if not children:
            return op_type

        # Sort children signatures for consistency
        child_sigs = sorted([structure_sig(child) for child in children])
        return f'{op_type}({",".join(child_sigs)})'

    return structure_sig(plan)
