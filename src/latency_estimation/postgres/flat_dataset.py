from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass
class FlatDatasetItem:
    query_id: str
    label: str
    latency: float
    features: np.ndarray
    plan: dict | None = None


class FlatArrayDataset(Dataset[FlatDatasetItem]):
    def __init__(self, items: list[FlatDatasetItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def x(self) -> np.ndarray:
        return np.vstack([item.features for item in self.items])

    def y(self) -> np.ndarray:
        return np.array([item.latency for item in self.items], dtype=np.float32)

    def plans(self) -> list[dict] | None:
        plans = [getattr(item, 'plan', None) for item in self.items]
        if any(plan is None for plan in plans):
            return None
        return plans


def save_flat_dataset(path: str, dataset: FlatArrayDataset) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(dataset, file)


def load_flat_dataset(path: str) -> FlatArrayDataset:
    with open(path, 'rb') as file:
        return pickle.load(file)
