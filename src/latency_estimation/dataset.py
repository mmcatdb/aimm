import json
import os
import pickle
from torch.utils.data import Dataset
from core.drivers import DriverType
from core.files import open_input, open_output
from core.nn_operator import NnOperator
from core.query import QueryInstanceId
from core.utils import print_warning
from .feature_extractor import PlanNode
from .model import BaseModel

DatasetName = str
"""Identifies a training dataset (within a specific driver).

Also used for related artifacts like extracted features.
"""

DatasetId = str
"""Identifies a training dataset.

Pattern: {driver_type}/{dataset_name}.
Example: "postgres/final_final_2".
"""

def create_dataset_id(driver_type: DriverType, dataset_name: DatasetName) -> DatasetId:
    return f'{driver_type.value}/{dataset_name}'

def parse_dataset_id(dataset_id: DatasetId) -> tuple[DriverType, DatasetName]:
    """Parses a dataset id into `driver_type`, `dataset_name`."""
    try:
        driver_type_str, dataset_name = dataset_id.split('/')
        return DriverType(driver_type_str), dataset_name
    except Exception as e:
        raise ValueError(f'Invalid dataset id: "{dataset_id}".') from e

class DatasetItem:

    def __init__(self, query_id: QueryInstanceId, label: str, latency: float, plan: PlanNode, structure_hash: int):
        self.query_id = query_id
        self.label = label
        self.latency = latency
        """In milliseconds."""
        self.plan = plan
        self.structure_hash = structure_hash

class ArrayDataset(Dataset[DatasetItem]):
    """
    Dataset of query plans with execution times. Implementation of PyTorch Dataset (there is no other implementation of this?).

    The items should contain something like:
    - query: The Cypher query string
    - plan: The query execution plan (from EXPLAIN)
    - execution_time: Actual measured execution time in ms
    """

    def __init__(self, items: list[DatasetItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

def prune_dataset(dataset: Dataset[DatasetItem], model: BaseModel) -> ArrayDataset:
    """Removes all query items that are not supported by the model (e.g., contain operators that are not in the model)."""
    pruned = set[str]()
    output = []

    for item in dataset:
        operators = model.find_missing_operators(item.plan)
        if operators:
            if item.query_id not in pruned:
                pruned.add(item.query_id)
                print_warning(f'Skipping query {item.query_id} because it contains unsupported operators: {", ".join([op.key() for op in operators])}.')
        else:
            output.append(item)

    return ArrayDataset(output)

def save_dataset(path: str, dataset: ArrayDataset):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump(dataset, file)

def load_dataset(path: str) -> ArrayDataset:
    with open(path, 'rb') as file:
        dataset = pickle.load(file)
        return dataset

def try_save_available_operators(path: str, operators: list[NnOperator]):
    try:
        with open_output(path) as file:
            sorted_operators = sorted(operators, key=lambda x: x.key())
            operators_data = [unit.to_dict() for unit in sorted_operators]
            json.dump(operators_data, file, indent=4)
    except Exception as e:
        print_warning(f'Could not save available operators to {path}.', e)

def try_load_available_operators(path: str) -> list[NnOperator] | None:
    try:
        with open_input(path) as file:
            operators_data = json.load(file)
            return [NnOperator.from_dict(data) for data in operators_data]
    except FileNotFoundError:
        print_warning(f'No available operators file found at {path}.')
        return None
    except Exception as e:
        print_warning(f'Could not load available operators from {path}.', e)
        return None
