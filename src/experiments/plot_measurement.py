import os
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from core.config import Config
from core.drivers import DriverType, DATABASE_COLORS
from experiments.measure import load_database_measurement
from scripts.analyze import get_subplots_dimensions

SCALES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
# TODO FIX mongo
# SCALES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

def main():
    config = Config.load()

    data = load_measurements(config)

    by_databases_path = os.path.join(config.results_directory, 'latency_by_databases.png')
    plot_by_databases(data, by_databases_path)

    by_queries_path = os.path.join(config.results_directory, 'latency_by_queries.png')
    plot_by_queries(data, by_queries_path)

ByQuery = dict[str, tuple[list[float], list[float]]]
ByDatabase = dict[DriverType, ByQuery]

def load_measurements(config: Config) -> ByDatabase:
    by_database = ByDatabase()
    for type in DriverType:
        by_database[type] = _load_database_measurement(config, type)

    return by_database

def _load_database_measurement(config: Config, type: DriverType) -> ByQuery:
    by_query = ByQuery()

    for scale in SCALES:
        info = DatabaseInfo(SCHEMA, type, scale)
        results = load_database_measurement(config, info)

        for result in results:
            id = result.id
            if id not in by_query:
                by_query[id] = ([], [])

            times = np.array(result.times)
            # errors = times.std() / np.sqrt(times.shape[0])
            errors = times.std()

            by_query[id][0].append(result.mean)
            by_query[id][1].append(errors)

    return by_query

def plot_by_databases(data: ByDatabase, save_path: str):
    plt.figure(figsize=(24, 10))
    for type in DriverType:
        db_data = data[type]
        plot_database_data(type, db_data)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {save_path}')

def plot_database_data(type: DriverType, data: ByQuery):
    all_types = [t for t in DriverType]
    plt.subplot(1, len(all_types), all_types.index(type) + 1)

    for query_id, (means, errors) in data.items():
        sizes = [scale_size_in_bytes(scale) for scale in SCALES]
        plt.errorbar(sizes, means, yerr=errors, label=query_id, capsize=5)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Dataset size [B]')
    plt.ylabel('Query latency [ms]')
    plt.title(f'Measurement Results - {type.value.capitalize()}')
    plt.legend()
    plt.tight_layout()

def plot_by_queries(data: ByDatabase, save_path: str):
    query_ids = data[DriverType.POSTGRES].keys()
    rows, cols = get_subplots_dimensions(len(query_ids))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if len(axes.shape) > 1 else [axes]

    fig.suptitle(f'Query Latency Comparison by Query', fontsize=16, fontweight='bold')

    for i, query_id in enumerate(query_ids):
        axis: Axes = axes[i]
        for type in DriverType:
            means, errors = data[type][query_id]
            sizes = [scale_size_in_bytes(scale) for scale in SCALES]
            axis.errorbar(sizes, means, yerr=errors, label=type.value.capitalize(), capsize=5, c=DATABASE_COLORS[type])

        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_xlabel('Dataset size [B]')
        axis.set_ylabel('Query latency [ms]')
        axis.set_title(f'Query {query_id}')
        axis.legend()
        plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {save_path}')

def scale_size_in_bytes(scale: float) -> int:
    """Approximate size of the dataset in bytes for a given scale factor."""
    a = 175530984.35196063
    b = 1.0476275776505792
    return a * scale ** b

# Fitted as:
# log_x = np.log(x)
# log_y = np.log(y)

# b, log_a = np.polyfit(log_x, log_y, 1)
# a = np.exp(log_a)

if __name__ == '__main__':
    main()
