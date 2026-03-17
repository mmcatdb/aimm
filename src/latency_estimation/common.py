from argparse import Namespace
from collections.abc import Callable
import os
import pickle
from typing import TypeVar
import torch
import numpy as np
from latency_estimation.abstract import BaseDataset
from common.utils import exit_with_error, print_warning, print_info

TDataset = TypeVar('TDataset', bound=BaseDataset)

def load_or_create_dataset(path: str | None, fallback: Callable[[], TDataset]) -> TDataset:
    if path is None:
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

    except FileNotFoundError:
        print_info(f'No cached dataset found at {path}, collecting new query plans...')
    except Exception as e:
        print_warning(f'Could not load cached dataset at {path}, collecting new query plans...', e)

    dataset = fallback()

    # Cache for future runs
    try:
        with open(path, 'wb') as file:
            pickle.dump(dataset, file)
            print(f'Collected and cached {len(dataset)} query plans')
    except Exception as e:
        print_warning(f'Could not cache dataset to {path}.', e)

    return dataset

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

def save_checkpoint_file(path: str, dict: dict, is_first_time: bool) -> None:
    if is_first_time and os.path.isfile(path):
        print_warning(f'Overwriting existing checkpoint file at {path}.')

    try:
        torch.save(dict, path)
    except Exception as e:
        # There is no point in continuing if we can't save the checkpoint.
        exit_with_error(f'Could not save checkpoint to {path}.', e)

def load_checkpoint_file(path: str, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        exit_with_error(f'Model checkpoint not found at {path}. Specify a valid --checkpoint path.')
    except Exception as e:
        exit_with_error(f'Could not load checkpoint from {path}.', e)

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
