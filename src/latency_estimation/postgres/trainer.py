from typing import cast
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from common.utils import EPSILON, print_warning
from latency_estimation.postgres.plan_extractor import PostgresItem
from latency_estimation.postgres.feature_extractor import FeatureExtractor
from latency_estimation.postgres.plan_structured_network import PlanStructuredNetwork

class PlanStructuredTrainer:
    """
    Trainer for plan-structured neural networks.
    Implements optimized training with batching and caching.
    """
    def __init__(self, model: PlanStructuredNetwork, learning_rate: float, batch_size: int):
        self.__model = model
        self.__optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        self.__loss_history: list[float] = []
        self.__batch_size = batch_size

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
            # Momentum should be saved automatically.
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }

    def evaluate(self, dataset: Dataset[PostgresItem]) -> dict[str, float]:
        """
        Evaluate model on a dataset.
        Returns:
            Dictionary of metrics (MAE, relative error, etc.)
        """
        self.__model.eval()

        estimations = []
        actuals = []

        with torch.no_grad():
            for item in dataset:
                try:
                    predicted_ms = self.__model(item.plan).item()
                except Exception as e:
                    print_warning(f'Could not compute model outputs for a query: \n{item.query}', e)
                    continue

                estimations.append(predicted_ms)
                actuals.append(item.execution_time)

        # Convert to numpy arrays
        estimations = np.array(estimations)
        actuals = np.array(actuals)

        # Mean absolute error
        mae = np.mean(np.abs(estimations - actuals)).item()
        # R-value: max(pred/actual, actual/pred)
        r_values = np.maximum(
            estimations / (actuals + EPSILON),
            actuals / (estimations + EPSILON)
        )
        # Mean relative error
        mre = np.mean(np.abs(estimations - actuals) / (actuals + EPSILON)).item()

        return {
            'mae': mae,
            'mre': mre,
            'mean_q_error': np.mean(r_values).item(),
            'median_q_error': np.median(r_values).item(),
            'le1.5_q_error': np.mean(r_values <= 1.5).item(),
            'le2.0_q_error': np.mean(r_values <= 2.0).item(),
        }

    @staticmethod
    def print_metrics(metrics: dict[str, float]):
        print('Evaluation Metrics:')
        print(f'  MAE: {metrics["mae"]:.2f} ms')
        print(f'  MRE: {metrics["mre"] * 100:.2f} %')
        print(f'  Mean R: {metrics["mean_q_error"]:.3f}')
        print(f'  Median R: {metrics["median_q_error"]:.3f}')
        print(f'  R ≤ 1.5: {metrics["le1.5_q_error"] * 100:.2f} %')
        print(f'  R ≤ 2.0: {metrics["le2.0_q_error"] * 100:.2f} %')
        print('')

    def train_epoch(self, dataset: Dataset[PostgresItem], batch_size: int | None = None, shuffle: bool = True) -> float:
        """
        Train for one epoch over the dataset.
        Args:
            dataset: Training dataset
            batch_size: Batch size
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
        Train on a single batch using plan-based batching.
        Args:
            batch: Batch of query plans
        Returns:
            Loss value for this batch
        """
        self.__model.train()
        self.__optimizer.zero_grad()

        # Group plans by structure
        groups = group_plans_by_structure(batch)

        # Compute gradient for each group
        total_loss = 0.0
        total_weight = 0

        for struct_hash, indexes in groups.items():
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
        avg_loss = total_loss / total_weight if total_weight > 0 else 0.0
        return avg_loss

    def __compute_loss(self, batch: list[dict]) -> torch.Tensor:
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
                all_outputs = self.__model.estimate_plan_latency_all_nodes(plan)
            except Exception as e:
                print_warning(f'Could not compute model outputs for a query: \n{item["query"]}', e)
                continue

            # Compute squared errors for all nodes
            for node_id, output_tensor in all_outputs.items():
                predicted_latency = output_tensor[0, 0] # Get latency (first element)

                if node_id in node_latencies:
                    actual_latency = node_latencies[node_id]

                    # TODO is the device needed?
                    actual_latency_tensor = torch.tensor(actual_latency, dtype=predicted_latency.dtype, device=predicted_latency.device)

                    squared_error = (predicted_latency - actual_latency_tensor) ** 2
                    total_squared_error += squared_error
                    total_nodes += 1

        # Return RMSE
        if total_nodes > 0:
            mse = cast(torch.Tensor, total_squared_error) / total_nodes
            # return torch.sqrt(mse)
            return torch.sqrt(mse + EPSILON)
        else:
            # Return a 0.0 tensor that still requires gradients
            return torch.tensor(0.0, requires_grad=True)

def group_plans_by_structure(batch: list[dict]) -> dict[str, list[int]]:
    """
    Group plans in a batch by their structure.
    Returns mapping from structure hash to indexes in batch.
    """
    groups = defaultdict(list)

    for index, item in enumerate(batch):
        plan = item['plan']
        struct_hash = compute_plan_structure_hash(plan)
        groups[struct_hash].append(index)

    return groups

def compute_plan_structure_hash(plan: dict) -> str:
    """
    Compute a hash representing the structure of a query plan.
    Plans with identical structure can be batched together.
    """
    def structure_sig(node):
        op_type = FeatureExtractor.get_node_type(node)
        children = FeatureExtractor.get_node_children(node)
        num_children = len(children)

        children_sig = []
        for child in children:
            children_sig.append(structure_sig(child))

        return f'{op_type}_{num_children}_{"_".join(sorted(children_sig))}'

    return structure_sig(plan)
