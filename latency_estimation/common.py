from argparse import Namespace
from collections.abc import Callable
import os
import pickle
import sys
from typing import TypeVar
import torch
import numpy as np
from latency_estimation.abstract import BaseDataset

class NnOperator:
    """Neural network node operator."""
    def __init__(self, type: str, num_children: int, feature_dim: int):
        self.type = type
        self.num_children = num_children
        self.feature_dim = feature_dim

    def key(self) -> str:
        """Get a unique key for this operator."""
        return f'{self.type}_{self.num_children}'

    def to_dict(self) -> dict:
        return {
            'type': self.type,
            'num_children': self.num_children,
            'feature_dim': self.feature_dim
        }

    @staticmethod
    def from_dict(data: dict) -> 'NnOperator':
        return NnOperator(
            type=data['type'],
            num_children=data['num_children'],
            feature_dim=data['feature_dim']
        )

TDataset = TypeVar('TDataset', bound = BaseDataset)

def load_dataset(path: str | None, fallback: Callable[[], TDataset]) -> TDataset:
    if (path is None):
        dataset = fallback()
        print(f'Collected {len(dataset)} query plans')
        return dataset

    # Try load cached dataset first.
    try:
        with open(path, 'rb') as file:
            dataset = pickle.load(file)
            print(f'Loaded {len(dataset)} cached query plans')
            # No need to cache again
            return dataset
    except Exception as e:
        print('No cached dataset found, collecting new query plans...')

    dataset = fallback()

    # Cache for future runs
    try:
        with open(path, 'wb') as file:
            pickle.dump(dataset, file)
            print(f'Collected and cached {len(dataset)} query plans')
    except Exception as e:
        print(f'Warning: Could not cache dataset to {path}: {e}')

    return dataset

def load_queries(args: Namespace, parser: Callable[[str], list[str]]) -> list[str]:
    if args.query:
        return [args.query]

    if not args.file:
        print('Error: Either --query or --file must be specified.', file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f'Error: File not found: {args.file}', file=sys.stderr)
        sys.exit(1)

    queries = parser(content)
    if not queries:
        print(f'Error: No queries found in {args.file}', file=sys.stderr)
        sys.exit(1)

    return queries

def save_checkpoint_file(path: str, dict: dict) -> None:
    if os.path.isfile(path):
        print(f'Warning: Overwriting existing checkpoint file at {path}', file=sys.stderr)

    try:
        torch.save(dict, path)
    except Exception as e:
        print(f'Error: Could not save checkpoint to {path}: {e}', file=sys.stderr)
        raise e

def load_checkpoint_file(path: str, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f'Error: Model checkpoint not found at {path}. Specify a valid --checkpoint path.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Error: Could not load checkpoint from {path}: {e}', file=sys.stderr)
        raise e

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

def format_latency(latency_seconds: float) -> str:
    """Format latency for human-readable output."""
    if latency_seconds is None:
        return 'ERROR'
    if latency_seconds < 0.001:
        return f'{latency_seconds * 1000000:.2f} µs'
    elif latency_seconds < 1:
        return f'{latency_seconds * 1000:.2f} ms'
    else:
        return f'{latency_seconds:.3f} s'

def truncate_query(query: str, max_length: int = 60) -> str:
    """Truncate query for display."""
    query = ' '.join(query.split())  # Normalize whitespace
    if len(query) > max_length:
        return query[:max_length - 3] + '...'
    return query

def print_dataset_summary(dataset: BaseDataset):
    print(f'  Total queries: {len(dataset)}')
    print(f'  Average execution time: {np.mean(dataset.execution_times):.2f} ms')
    print(f'  Min execution time: {np.min(dataset.execution_times):.2f} ms')
    print(f'  Max execution time: {np.max(dataset.execution_times):.2f} ms')
    print(f'  Median execution time: {np.median(dataset.execution_times):.2f} ms')
