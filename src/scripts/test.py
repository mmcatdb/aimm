import os
import argparse
import json
from core.config import Config
from core.utils import JsonEncoder, exit_with_exception, print_warning
from latency_estimation.class_provider import get_model_evaluator
from latency_estimation.dataset import create_dataset_id, load_dataset
from latency_estimation.model import parse_checkpoint_id
from providers.contex import Context

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Test models on queries.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    ctx = Context.default()

    try:
        run(ctx, args)
    except Exception as e:
        exit_with_exception(e)

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('checkpoint_id', nargs=1, help='Id of the model checkpoint. Pattern: {driver_type}/{model_name}/{checkpoint_name}.')
    parser.add_argument('test_dataset',  nargs=1, help='Name of the test dataset.')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots. Only for supported drivers (e.g. postgres).')

def run(ctx: Context, args: argparse.Namespace):
    checkpoint_id = args.checkpoint_id[0]
    dataset_name = args.test_dataset[0]

    driver_type, _, _ = parse_checkpoint_id(checkpoint_id)

    test_id = create_dataset_id(driver_type, dataset_name)
    test_dataset = load_dataset(ctx.pp.dataset(test_id))

    model = ctx.mp.load_model(ctx.pp.model(checkpoint_id))
    evaluator = get_model_evaluator(driver_type, model)

    results = evaluator.evaluate_dataset(test_dataset)
    evaluator.print_summary(results)

    save_results(ctx.config, results)

    if not args.no_plots:
        try:
            plot_path = os.path.join(ctx.config.results_directory, 'evaluation_plots.png')
            evaluator.plot_results(results, save_path=plot_path)
        except Exception as e:
            print_warning('Could not generate plots.', e)

def save_results(config: Config, results: list):
    path = os.path.join(config.results_directory, 'evaluation_results.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f'\nSaving results to {path}...')
    with open(path, 'w') as file:
        json.dump(results, file, indent=4, cls=JsonEncoder)

if __name__ == '__main__':
    main()
