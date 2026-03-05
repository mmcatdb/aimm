import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from plan_structured_network import PlanStructuredNetwork


class QueryPlanDataset:
    """Dataset of MongoDB query plans with execution times."""

    def __init__(self, samples: List[Dict], network: PlanStructuredNetwork):
        """
        Args:
            samples: list of dicts from PlanExtractor.collect_training_data
            network: PlanStructuredNetwork (needed for plan tree extraction)
        """
        self.items = []
        for s in samples:
            plan_tree = network.extract_plan_tree(s["explain"])
            self.items.append({
                "plan_tree": plan_tree,
                "collection": s["collection"],
                "execution_time_ms": s["execution_time_ms"],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class PlanStructuredTrainer:
    """
    Trainer for MongoDB plan-structured neural networks.

    Loss: log-latency MSE  L = (log(pred+eps) - log(actual+eps))^2
    This optimizes for geometric mean relative error.
    """

    def __init__(self, model: PlanStructuredNetwork,
                 learning_rate: float = 0.001,
                 warmup_epochs: int = 10,
                 total_epochs: int = 200):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=learning_rate * 0.01
        )
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = learning_rate
        self.loss_history = []
        self.eps = 0.0001  # small constant to avoid log(0)
        self._epoch = 0

    def compute_loss(self, batch: List[Dict]) -> torch.Tensor:
        """
        Compute log-latency MSE loss over a batch.
        Only uses root-level latency prediction vs actual execution time.
        """
        losses = []
        for item in batch:
            plan_tree = item["plan_tree"]
            collection = item["collection"]
            actual_ms = item["execution_time_ms"]

            try:
                pred_ms = self.model(plan_tree, collection)
            except Exception as e:
                continue

            log_pred = torch.log(pred_ms + self.eps)
            log_actual = torch.log(
                torch.tensor(actual_ms + self.eps,
                             dtype=log_pred.dtype,
                             device=log_pred.device)
            )
            loss = (log_pred - log_actual) ** 2
            losses.append(loss)

        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, requires_grad=True)

    def train_batch(self, batch: List[Dict]) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, dataset: QueryPlanDataset,
                    batch_size: int = 32) -> float:
        indices = np.random.permutation(len(dataset))
        epoch_losses = []

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [dataset[idx] for idx in batch_indices]
            loss = self.train_batch(batch)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        self.loss_history.append(avg_loss)

        # Step scheduler after warmup
        self._epoch += 1
        if self._epoch > self.warmup_epochs:
            self.scheduler.step()

        return avg_loss

    def evaluate(self, dataset: QueryPlanDataset) -> Dict[str, float]:
        """Evaluate model on a dataset, returning standard QPP metrics."""
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for item in dataset:
                plan_tree = item["plan_tree"]
                collection = item["collection"]
                actual_ms = item["execution_time_ms"]

                try:
                    pred_ms = self.model(plan_tree, collection).item()
                except Exception:
                    continue

                predictions.append(pred_ms)
                actuals.append(actual_ms)

        if not predictions:
            return {"mae": float("inf"), "relative_error": float("inf")}

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        relative_error = np.mean(np.abs(predictions - actuals) / (actuals + 1e-8))

        # Q-error (R-value)
        r_values = np.maximum(
            predictions / (actuals + 1e-8),
            actuals / (predictions + 1e-8),
        )

        # Geometric mean of Q-error
        log_qerror = np.mean(np.log(r_values + 1e-8))

        metrics = {
            "mae": mae,
            "relative_error": relative_error,
            "median_r": np.median(r_values),
            "mean_r": np.mean(r_values),
            "geo_mean_r": np.exp(log_qerror),
            "r_within_1.5": np.mean(r_values <= 1.5),
            "r_within_2.0": np.mean(r_values <= 2.0),
            "r_within_5.0": np.mean(r_values <= 5.0),
        }
        return metrics
