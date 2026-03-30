from genericpath import isfile
import math
import os
import argparse
import re
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from common.config import Config
from common.drivers import DriverType
from common.utils import exit_with_error, print_warning
from common.database import DatabaseInfo
from datasets.databases import TRAIN_DATASET
from latency_estimation.trainer import TrainerMetrics
from latency_estimation.context import BaseContext, load_metrics_file, load_checkpoint_file

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Loads metrics from json files and plots them into a graph.')

    parser.add_argument('database', nargs=1, choices=[dataset.value for dataset in DriverType], help=f'Type of database to analyze.')

    args = parser.parse_args(rawArgs)
    type = DriverType(args.database[0])
    info = DatabaseInfo(TRAIN_DATASET, type, None)

    _run(info)

def _run(info: DatabaseInfo):
    config = Config.load()

    metrics_paths = _get_valid_file_paths(config.metrics_directory, BaseContext.metrics_epoch_filename_regex(info))
    if not metrics_paths:
        exit_with_error(f'No valid metrics files found for {info.id()} in {config.metrics_directory}.')

    print(f'Found {len(metrics_paths)} metrics files for {info.id()}.')

    epochs = list[int]()
    metrics = list[TrainerMetrics]()

    for (epoch, path) in metrics_paths:
        epoch_metrics = load_metrics_file(path)
        if epoch_metrics:
            epochs.append(epoch)
            metrics.append(epoch_metrics)

    losses = list[float]()

    checkpoint_paths = _get_valid_file_paths(config.checkpoints_directory, BaseContext.checkpoint_epoch_filename_regex(info))
    if not checkpoint_paths:
        print_warning(f'No valid checkpoint files found for {info.id()} in {config.checkpoints_directory}.')
    else:
        _, checkpoint_path = checkpoint_paths[-1]
        checkpoint = load_checkpoint_file(checkpoint_path, 'cpu') # We don't care about the device here, we just want to get the losses.
        # TODO This is not ideal - use some unified accessor from the trainer instead.
        losses = BaseContext.get_loss_history_from_checkpoint(checkpoint)

    title = f'{info.label()} Metrics over {max(epochs)} Epochs'
    save_path = os.path.join(config.results_directory, f'{info.id()}_metrics.png')

    _plot_metrics_and_losses(epochs, metrics, losses, title, save_path)

def _get_valid_file_paths(path: str, regex: re.Pattern) -> list[tuple[int, str]]:
    output = list[tuple[int, str]]()

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if not isfile(filepath):
            continue

        match = regex.fullmatch(filename)
        if not match:
            continue

        epoch = int(match.group(1))
        output.append((epoch, filepath))

    output.sort(key=lambda x: x[0]) # Sort by epoch epoch

    return output

def _plot_metrics_and_losses(epochs: list[int], metrics: list[TrainerMetrics], losses: list[float], suptitle: str, save_path: str):
    metrics_keys = list(metrics[0].keys())
    num_plots = len(metrics_keys) + (1 if losses else 0)

    rows, cols = get_subplots_dimensions(num_plots)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if len(axes.shape) > 1 else [axes]

    fig.suptitle(suptitle, fontsize=16, fontweight='bold')

    for i in range(len(metrics_keys)):
        key = metrics_keys[i]
        metric_values = [m[key] for m in metrics]
        _plot_subplot(axes[i], epochs, metric_values, key)
        plt.tight_layout()

    if losses:
        loss_epochs = list(range(1, len(losses) + 1))
        _plot_subplot(axes[len(metrics_keys)], loss_epochs, losses, 'loss')
        plt.tight_layout()

    plt.savefig(save_path)
    print(f'Metrics plot saved to {save_path}.')

def get_subplots_dimensions(num_plots: int):
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)
    return rows, cols

def _plot_subplot(axis: Axes, epochs: list[int], values: list[float], key: str):
    axis.plot(epochs, values, label=key, marker='o')
    axis.set_xlabel('Epoch')
    axis.set_ylabel(key)
    axis.set_title(f'{key} over Epochs')
    axis.legend()
    axis.grid()

if __name__ == '__main__':
    main()
