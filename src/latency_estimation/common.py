from argparse import Namespace
from collections.abc import Callable
import pickle
from typing import Generic, TypeVar
from common.utils import exit_with_error, print_warning, print_info, time_quantity
from torch.utils.data import Dataset
from latency_estimation.feature_extractor import BaseDatasetItem
from latency_estimation.plan_structured_network import BasePlanStructuredNetwork

TDatasetItem = TypeVar('TDatasetItem', bound=BaseDatasetItem)

class ArrayDataset(Dataset[TDatasetItem]):
    """
    Dataset of query plans with execution times. Implementation of PyTorch Dataset (there is no other implementation of this?).

    The items should contain something like:
    - query: The Cypher query string
    - plan: The query execution plan (from EXPLAIN)
    - execution_time: Actual measured execution time in ms
    """
    def __init__(self, items: list[TDatasetItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

TDataset = TypeVar('TDataset', bound=Dataset)

class DatasetBundle(Generic[TDatasetItem]):
    """Container for all datasets related to latency estimation."""
    def __init__(self, train: ArrayDataset[TDatasetItem], val: ArrayDataset[TDatasetItem]):
        self.train = train
        self.val = val
        # self.test = test

    def length(self):
        # The length is missing on the base Dataset class, but it's present on all our implementations as well as ConcatDataset and others.
        # So we can safely cast to Sized to get the length.
        # Ok, we are not casting it here because we use the ArrayDataset ... but if we ever switch to the base Dataset, we could use this ...
        return len(self.train) + len(self.val) # + len(self.test)

def load_or_create_dataset(path: str | None, fallback: Callable[[], DatasetBundle[TDatasetItem]]) -> DatasetBundle[TDatasetItem]:
    if path is None:
        dataset = fallback()
        print(f'Collected {dataset.length()} query plans')
        return dataset

    # Try load cached dataset first.
    try:
        with open(path, 'rb') as file:
            dataset = pickle.load(file)
            print(f'Loaded {dataset.length()} cached query plans')

            # No need to cache again
            return dataset

    except FileNotFoundError:
        print_info(f'No cached dataset found at {path}, collecting new query plans...')
    except Exception as e:
        print_warning(f'Could not load cached dataset at {path}, collecting new query plans...', e)

    dataset = fallback()

    # Cache for future runs
    try:
        with open(path, 'wb') as file:
            pickle.dump(dataset, file)
            print(f'Collected and cached {dataset.length()} query plans')
    except Exception as e:
        print_warning(f'Could not cache dataset to {path}.', e)

    return dataset

def prune_dataset(dataset: Dataset[TDatasetItem], model: BasePlanStructuredNetwork) -> ArrayDataset[TDatasetItem]:
    """Removes all query items that are not supported by the model (e.g., contain operators that are not in the model)."""
    pruned = set[str]()
    output = []

    for item in dataset:
        operators = model.find_missing_operators(item.plan)
        if operators:
            query_string = item.query_string()
            if (query_string not in pruned):
                pruned.add(query_string)
                print_warning(f'Skipping query because it contains unsupported operators: {", ".join([op.key() for op in operators])}.\n{truncate_query(query_string)}\n')
        else:
            output.append(item)

    return ArrayDataset(output)

def load_queries(args: Namespace, parser: Callable[[str], list[str]]) -> list[str]:
    if args.query:
        return [args.query]

    if not args.file:
        exit_with_error('Either --query or --file must be specified.')

    try:
        with open(args.file, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        exit_with_error(f'File not found: {args.file}. Specify a valid --file path.')

    queries = parser(content)
    if not queries:
        exit_with_error(f'No queries found in {args.file}.')

    return queries

def parse_queries(content: str) -> list[str]:
    """Works for both PostgreSQL and Neo4j query files."""
    # Remove comments
    lines = []
    for line in content.split('\n'):
        # Remove inline comments
        if '#' in line:
            line = line[:line.index('#')]
        if '--' in line:
            line = line[:line.index('--')]
        lines.append(line)

    content = '\n'.join(lines)

    # Split by semicolons if present, otherwise by double newlines
    if ';' in content:
        queries = [q.strip() for q in content.split(';')]
    else:
        # Try splitting by double newlines for multi-line queries
        queries = [q.strip() for q in content.split('\n\n')]

        # If that results in single lines, treat each non-empty line as a query
        if all('\n' not in q for q in queries if q):
            queries = [q.strip() for q in content.split('\n')]

    # Filter out empty queries
    queries = [q for q in queries if q.strip()]

    return queries

def format_latency(latency_ms: float) -> str:
    """Format latency for human-readable output."""
    return time_quantity.pretty_print(latency_ms)

def truncate_query(query: str, max_length: int = 60) -> str:
    """Truncate query for display."""
    query = ' '.join(query.split())  # Normalize whitespace
    if len(query) > max_length:
        return query[:max_length - 3] + '...'
    return query
