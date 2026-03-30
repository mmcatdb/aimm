import os
from matplotlib import pyplot as plt, ticker
import numpy as np
from scipy.optimize import curve_fit
from common.config import Config, DatasetName
from experiments.mcts import find_stats, load_stats, IterationStats, LoadedAdaptationSolution

DATASET = DatasetName.EDBT

def main():
    config = Config.load()

    scales = list[float]()
    iteration_sets = list[list[IterationStats]]()
    solution_sets = list[list[LoadedAdaptationSolution]]()

    runs = find_stats(config)
    for scale, run_id in runs:
        if scale != 1:
            continue

        scales.append(scale)
        iterations, solutions = load_stats(config, run_id, scale)
        iteration_sets.append(iterations)
        solution_sets.append(solutions)

    plot_iterations(config, scales, iteration_sets)

    plot_solutions(config, solution_sets)

IterationSets = list[list[IterationStats]]

def plot_iterations(config: Config, scales: list[float], iteration_sets: IterationSets):
    counts_by_scale = dict[float, IterationSeries]()
    states_by_scale = dict[float, IterationSeries]()

    for i, scale in enumerate(scales):
        set_ = iteration_sets[i]

        if scale not in counts_by_scale:
            counts_by_scale[scale] = IterationSeries()
            states_by_scale[scale] = IterationSeries()

        times = [stats[0] for stats in set_]
        counts = [stats[1] for stats in set_]
        states = [stats[2] for stats in set_]

        counts_by_scale[scale].append((times, counts))
        states_by_scale[scale].append((times, states))

    counts_path = os.path.join(config.results_directory, 'mcts_counts.png')
    plot_iteration_series(counts_by_scale, 'MCTS iterations', counts_path)

    states_path = os.path.join(config.results_directory, 'mcts_states.png')
    plot_iteration_series(states_by_scale, 'Unique states', states_path)

IterationSeries = list[tuple[list[float], list[int]]]

def plot_iteration_series(series_by_scale: dict[float, IterationSeries], label: str, save_path: str):
    plt.figure(figsize=(6, 4))

    for scale, series in series_by_scale.items():
        _plot_x_mean_line(series, label=f'{label} ({scale:g})')

    plt.xlabel('Time [s]')
    plt.ylabel(label)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', scilimits=(-3, 3))
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {save_path}')

def _plot_x_mean_line(series: IterationSeries, label: str):
    y_min = min(min(y) for _, y in series)
    y_max = max(max(y) for _, y in series)

    common_y = np.linspace(y_min, y_max, 100)
    xs = []

    for (x, y) in series:
        x_interp = np.interp(common_y, y, x)
        xs.append(x_interp)

        # Plot individual runs with low opacity
        plt.plot(x, y, color='gray', alpha=0.3, linewidth=1)

    x_np = np.array(xs)
    x_mean = x_np.mean(axis=0)
    plt.plot(x_mean, common_y, label=label, color='red')

def plot_solutions(config: Config, solution_sets: list[list[LoadedAdaptationSolution]]):
    plt.figure(figsize=(6, 4))

    all_x = list[list[float]]()
    all_y = list[list[float]]()

    for solutions in solution_sets:
        times = _normalize_times([solution.found_in for solution in solutions])
        latencies = [solution.latency for solution in solutions]
        initial_latency = latencies[0]
        costs = [latency / initial_latency for latency in latencies]

        all_x.append(times)
        all_y.append(costs)

        # Plot individual runs with low opacity
        plt.plot(times, costs, color='gray', alpha=0.3, linewidth=1)

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    # log(1 + x)
    # model: anchored at y(0) = 1
    # def model(x, b):
    #     return 1 + b * np.log1p(x)

    # params, _ = curve_fit(model, all_x, all_y, maxfev=10000)

    # def model(x, a, b):
    #     return 1 / (1 + a * x**b)

    # params, _ = curve_fit(model, all_x, all_y, p0=[0.01, 0.5], maxfev=10000)

    def model(x, a, b, c):
        return c + (1 - c) / (1 + a * x**b)

    params, _ = curve_fit(model, all_x, all_y, maxfev=10000)

    x_fit = np.linspace(min(all_x), max(all_x), 200)
    y_fit = model(x_fit, *params)

    plt.plot(x_fit, y_fit, color='red', label='Log fit')

    plt.xlabel('Time [s]')
    plt.xscale('log')
    plt.ylabel('Relative workload cost')
    plt.tight_layout()

    save_path = os.path.join(config.results_directory, 'mcts_solutions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {save_path}')

def _normalize_times(times: list[float]) -> list[float]:
    prev = times[0]
    normalized = []
    for time in times:
        while time <= prev:
            time += 0.001
        normalized.append(time)
        prev = time

    return normalized

if __name__ == '__main__':
    main()
