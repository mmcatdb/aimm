from __future__ import annotations

import os
import pickle
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset


@dataclass
class MongoFlatDatasetItem:
    query_id: str
    label: str
    latency: float
    features: np.ndarray
    plan: dict | None = None
    global_stats: dict | None = None


class MongoFlatArrayDataset(Dataset[MongoFlatDatasetItem]):
    def __init__(self, items: list[MongoFlatDatasetItem]):
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

    def plan_samples(self) -> list[tuple[dict, dict]] | None:
        samples = []
        for item in self.items:
            if item.plan is None or item.global_stats is None:
                return None
            samples.append((item.plan, item.global_stats))
        return samples


def save_mongo_flat_dataset(path: str, dataset: MongoFlatArrayDataset) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(dataset, file)


def load_mongo_flat_dataset(path: str) -> MongoFlatArrayDataset:
    with open(path, 'rb') as file:
        return pickle.load(file)
