"""
Training pipeline for plan-structured neural networks.
"""
from typing import cast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from plan_structured_network import PlanStructuredNetwork


class QueryPlanDataset(Dataset):
    """Dataset of query plans with execution times."""

    def __init__(self, plans: list[dict], execution_times: list[float]):
        """
        Args:
            plans: List of query plan dictionaries
            execution_times: List of execution times (ground truth)
        """
        self.plans = plans
        self.execution_times = execution_times

        # Extract actual latencies for all nodes in all plans
        self.node_latencies = []
        for plan, total_time in zip(plans, execution_times):
            node_times = self._extract_node_latencies(plan)
            self.node_latencies.append(node_times)

    def _extract_node_latencies(self, node: dict) -> dict[int, float]:
        """
        Extract actual execution time for each node in the plan.
        PostgreSQL provides this via EXPLAIN ANALYZE.
        """
        latencies = {}

        def traverse(n):
            # Actual time is in format [start, end]
            if 'Actual Total Time' in n:
                latencies[id(n)] = n['Actual Total Time']
            else:
                # Fallback: use a portion of total time
                latencies[id(n)] = 0.0

            if 'Plans' in n:
                for child in n['Plans']:
                    traverse(child)

        traverse(node)
        return latencies

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, idx):
        return {
            'plan': self.plans[idx],
            'execution_time': self.execution_times[idx],
            'node_latencies': self.node_latencies[idx]
        }


def compute_plan_structure_hash(plan: dict) -> str:
    """
    Compute a hash representing the structure of a query plan.
    Plans with identical structure can be batched together.
    """
    def structure_sig(node):
        node_type = node.get('Node Type', '')
        num_children = len(node.get('Plans', []))

        children_sig = []
        if 'Plans' in node:
            for child in node['Plans']:
                children_sig.append(structure_sig(child))

        return f'{node_type}_{num_children}_{"_".join(sorted(children_sig))}'

    return structure_sig(plan)


def group_plans_by_structure(batch: list[dict]) -> dict[str, list[int]]:
    """
    Group plans in a batch by their structure.
    Returns mapping from structure hash to indices in batch.
    """
    groups = defaultdict(list)

    for idx, item in enumerate(batch):
        plan = item['plan']
        struct_hash = compute_plan_structure_hash(plan)
        groups[struct_hash].append(idx)

    return groups


class PlanStructuredTrainer:
    """
    Trainer for plan-structured neural networks.
    Implements optimized training with batching and caching.
    """

    def __init__(self, model: PlanStructuredNetwork, learning_rate: float = 0.001, momentum: float = 0.9):
        """
        Args:
            model: Plan-structured neural network
            learning_rate: Learning rate for SGD
            momentum: Momentum for SGD
        """
        self.model = model
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum
        )
        self.loss_history: list[float] = []


    def compute_loss(self, batch: list[dict]) -> torch.Tensor:
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
            plan = item['plan']
            node_latencies = item['node_latencies'] # {node_id: actual_latency}

            try:
                all_outputs = self.model.get_all_node_predictions(plan)
            except Exception as e:
                print(f'Error processing plan: {e}. Skipping plan.')
                continue

            # Compute squared errors for all nodes
            for node_id, output_tensor in all_outputs.items():
                pred_latency = output_tensor[0, 0] # Get latency (first element)

                if node_id in node_latencies:
                    actual_latency = node_latencies[node_id]

                    actual_latency_tensor = torch.tensor(actual_latency, dtype=pred_latency.dtype, device=pred_latency.device)

                    squared_error = (pred_latency - actual_latency_tensor) ** 2
                    total_squared_error += squared_error
                    total_nodes += 1

        # Return RMSE
        if total_nodes > 0:
            mse = cast(torch.Tensor, total_squared_error) / total_nodes
            # return torch.sqrt(mse)
            return torch.sqrt(mse + 1e-8)
        else:
            # Return a 0.0 tensor that still requires gradients
            return torch.tensor(0.0, requires_grad=True)

    def train_batch(self, batch: list[dict]) -> float:
        """
        Train on a single batch using plan-based batching.

        Args:
            batch: Batch of query plans

        Returns:
            Loss value for this batch
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Group plans by structure
        groups = group_plans_by_structure(batch)

        # Compute gradient for each group
        total_loss = 0.0
        total_weight = 0

        for struct_hash, indices in groups.items():
            group_batch = [batch[i] for i in indices]
            group_size = len(group_batch)

            # Compute loss for this group
            loss = self.compute_loss(group_batch)

            # Weighted by group size (for proper gradient estimation)
            weighted_loss = loss * group_size
            total_loss += weighted_loss.item()
            total_weight += group_size

            # Backward pass (accumulates gradients)
            weighted_loss.backward()

        # Normalize gradients by total batch size
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad /= total_weight

        # Update weights
        self.optimizer.step()

        # Return average loss
        avg_loss = total_loss / total_weight if total_weight > 0 else 0.0
        return avg_loss

    def train_epoch(self, dataset: QueryPlanDataset, batch_size: int = 32) -> float:
        """
        Train for one epoch over the dataset.

        Args:
            dataset: Training dataset
            batch_size: Batch size

        Returns:
            Average loss for the epoch
        """
        # Shuffle dataset
        indices = np.random.permutation(len(dataset))

        epoch_losses = []

        # Process in batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [dataset[idx] for idx in batch_indices]

            loss = self.train_batch(batch)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses).item()
        self.loss_history.append(avg_loss)

        return avg_loss

    def evaluate(self, dataset: QueryPlanDataset) -> dict[str, float]:
        """
        Evaluate model on a dataset.

        Returns:
            Dictionary of metrics (MAE, relative error, etc.)
        """
        self.model.eval()

        predictions = []
        actuals = []

        with torch.no_grad():
            for item in dataset:
                plan = item['plan']
                actual_time = item['execution_time']

                # Predict
                pred_time = self.model(plan).item()

                predictions.append(pred_time)
                actuals.append(actual_time)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Compute metrics
        mae = np.mean(np.abs(predictions - actuals)).item()
        relative_error = np.mean(np.abs(predictions - actuals) / (actuals + 1e-8)).item()

        # R(q) metric
        r_values = np.maximum(
            predictions / (actuals + 1e-8),
            actuals / (predictions + 1e-8)
        )

        metrics = {
            'mae': mae,
            'relative_error': relative_error,
            'median_r': np.median(r_values).item(),
            'r_within_1.5': np.mean(r_values <= 1.5).item(),
            'r_within_2.0': np.mean(r_values <= 2.0).item(),
        }

        return metrics
