import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
from common.utils import EPSILON
from latency_estimation.abstract import BaseDataset
from latency_estimation.neo4j.plan_structured_network import PlanStructuredNetwork

class PlanStructuredTrainer:
    """
    Trainer for plan-structured neural networks (Neo4j version).
    Implements optimized training with batching and caching.
    """
    def __init__(self, model: PlanStructuredNetwork, learning_rate: float, batch_size: int):
        self.__model = model
        self.__optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.__loss_history: list[float] = []
        self.__batch_size = batch_size

        # Loss function: MSE for total query latency
        self.criterion = nn.MSELoss()

    @staticmethod
    def load_from_checkpoint(model: PlanStructuredNetwork, checkpoint: dict, learning_rate: float, batch_size: int) -> 'PlanStructuredTrainer':
        trainer = PlanStructuredTrainer(model, learning_rate, batch_size)
        trainer.__optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.__loss_history = checkpoint['loss_history']
        return trainer

    def to_checkpoint(self) -> dict:
        """Serialization to a file-friendly dictionary."""
        return {
            'epoch': len(self.__loss_history),
            'loss_history': self.__loss_history,
            # We don't save the learning rate as it can be changed between sessions.
            # Weight decay should be saved automatically.
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }

    def evaluate(self, dataset: BaseDataset[str], batch_size: int | None = None) -> dict[str, float]:
        """
        Evaluate model on a dataset.
        Returns:
            Dictionary with evaluation metrics
        """
        batch_size = batch_size if batch_size is not None else self.__batch_size

        self.__model.eval()

        # Identity collate_fn - return list of items as-is
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

        estimations = []
        actuals = []

        with torch.no_grad():
            for batch in dataloader:
                for item in batch:
                    estimated_latency = self.__model(item['plan']).item()
                    estimations.append(estimated_latency)
                    actuals.append(item['execution_time'])

        # Convert to numpy arrays
        estimations = np.array(estimations)
        actuals = np.array(actuals)

        # Mean squared error
        mse = np.mean((estimations - actuals) ** 2).item()
        rmse = np.sqrt(mse)
        # Mean absolute error
        mae = np.mean(np.abs(estimations - actuals)).item()
        # R-value: max(pred/actual, actual/pred)
        r_values = np.maximum(
            estimations / (actuals + EPSILON),
            actuals / (estimations + EPSILON)
        )
        # Mean relative error
        mre = np.mean(np.abs(estimations - actuals) / (actuals + EPSILON)).item()

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mre': mre,
            'mean_q_error': np.mean(r_values).item(),
            'median_q_error': np.median(r_values).item(),
            'p90_q_error': np.percentile(r_values, 90).item(),
            'p95_q_error': np.percentile(r_values, 95).item(),
        }

        return metrics

    @staticmethod
    def print_metrics(metrics: dict[str, float]):
        print('Evaluation Metrics:')
        print(f'  MAE: {metrics["mae"]:.2f} ms')
        print(f'  MRE: {metrics["mre"] * 100:.2f} %')
        print(f'  RMSE: {metrics["rmse"]:.2f} ms')
        print(f'  Mean R: {metrics["mean_q_error"]:.3f}')
        print(f'  Median R: {metrics["median_q_error"]:.3f}')
        print(f'  P90 R: {metrics["p90_q_error"] * 100:.2f} %')
        print(f'  P95 R: {metrics["p95_q_error"] * 100:.2f} %')
        print('')

    def train_epoch(self, dataset: BaseDataset[str], batch_size: int | None = None, shuffle: bool = True) -> float:
        """
        Args:
            dataset: Training dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
        Returns:
            Average loss for the epoch
        """
        batch_size = batch_size if batch_size is not None else self.__batch_size
        # Identity collate_fn - return list of items as-is
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x)

        epoch_losses = []

        for index, batch in enumerate(dataloader):
            loss = self.__train_batch(batch)
            epoch_losses.append(loss)

            if (index + 1) % 10 == 0:
                print(f'  Batch {index + 1}/{len(dataloader)}, Loss: {loss:.6f}')

        avg_loss = np.mean(epoch_losses).item()
        self.__loss_history.append(avg_loss)

        return avg_loss

    def __train_batch(self, batch: list[dict]) -> float:
        """
        Args:
            batch: List of batch items
        Returns:
            Loss value for this batch
        """
        self.__model.train()
        self.__optimizer.zero_grad()

        # Compute loss
        loss = self.__compute_loss(batch)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.__model.parameters(), max_norm=1.0)

        # Update weights
        self.__optimizer.step()

        return loss.item()

    def __compute_loss(self, batch: list[dict]) -> torch.Tensor:
        """
        Compute MSLE loss for a batch of query plans.
        Loss = (log(estimated + 1) - log(actual + 1))²
        """
        estimations = []
        targets = []

        # Group plans by structure for efficiency
        structure_groups = group_plans_by_structure(batch)

        for structure_hash, indexes in structure_groups.items():
            # Process plans with same structure together
            for index in indexes:
                plan = batch[index]['plan']
                execution_time = batch[index]['execution_time']

                # Forward pass through model
                estimated_latency = self.__model(plan)

                estimations.append(estimated_latency)
                targets.append(torch.tensor([[execution_time]], dtype=torch.float32))
                # TODO not sure about this ... is the device needed?
                # targets.append(torch.tensor([[execution_time]], dtype=torch.float32, device=self.device))

        # Stack estimations and targets
        estimations = torch.cat(estimations, dim=0)
        targets = torch.cat(targets, dim=0)

        # Apply Log transformation before MSE
        log_preds = torch.log1p(torch.abs(estimations))
        log_targets = torch.log1p(targets)

        # Compute MSE on log values -> MSLE
        return self.criterion(log_preds, log_targets)

def group_plans_by_structure(batch: list[dict]) -> dict[str, list[int]]:
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
        plan = item['plan']
        structure = compute_plan_structure_hash(plan)
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
        op_type = node.get('operatorType', 'Unknown').replace('@neo4j', '')
        children = node.get('children', [])

        if not children:
            return op_type

        # Sort children signatures for consistency
        child_sigs = sorted([structure_sig(child) for child in children])
        return f'{op_type}({",".join(child_sigs)})'

    return structure_sig(plan)
