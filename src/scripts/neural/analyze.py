from genericpath import isfile
import math
import os
import argparse
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from core.config import Config
from core.utils import exit_with_error, exit_with_exception
from latency_estimation.model import ModelId
from latency_estimation.trainer import BaseTrainer, TrainerMetrics, load_metrics
from providers.path_provider import PathProvider

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Loads metrics from json files and plots them into a graph.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    config = Config.load()
    model_id = args.model_id[0]

    try:
        run(config, model_id)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('model_id', nargs=1, help='Id of the model. Pattern: {driver_type}/{model_name}')

def run(config: Config, model_id: ModelId):
    pp = PathProvider(config)
    epochs_directory = pp.epochs(model_id)

    metrics_paths = _get_valid_file_paths(epochs_directory, pp.METRICS_SUFFIX)
    if not metrics_paths:
        exit_with_error(f'No valid metrics files found for {model_id} in {metrics_paths}.')

    print(f'Found {len(metrics_paths)} metrics files for {model_id}.')

    metrics = [load_metrics(path) for path in metrics_paths]
    metrics.sort(key=lambda m: BaseTrainer.get_epoch_and_loss_from_metrics(m)[0]) # Sort by epoch

    epochs = list[int]()
    losses = list[float]()
    for m in metrics:
        epoch, loss = BaseTrainer.get_epoch_and_loss_from_metrics(m)
        epochs.append(epoch)
        losses.append(loss)

    title = f'{model_id} Metrics over {max(epochs)} Epochs'
    # TODO path provider for this as well?
    save_path = os.path.join(config.results_directory, f'{model_id}_metrics.png')

    _plot_metrics_and_losses(epochs, metrics, losses, title, save_path)

def _get_valid_file_paths(path: str, suffix: str) -> list[str]:
    output = list[str]()

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if isfile(filepath) and filename.endswith(suffix):
            output.append(filepath)

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

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
