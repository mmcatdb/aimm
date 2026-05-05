import argparse
import json
import os

from core.config import Config
from core.drivers import DriverType
from core.utils import JsonEncoder, exit_with_exception
from latency_estimation.dataset import create_dataset_id, parse_dataset_id
from latency_estimation.postgres.flat_dataset import load_flat_dataset
from latency_estimation.postgres.flat_model import compute_metrics, load_flat_model, print_metrics, transform_dataset_features
from providers.path_provider import PathProvider


def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Evaluate a PostgreSQL flat-feature latency model on a flat dataset.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    try:
        run(Config.load(), args)
    except Exception as e:
        exit_with_exception(e)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('model_id', nargs=1, help='Id of the model. Pattern: postgres/{model_name}.')
    parser.add_argument('test_dataset', nargs=1, help='Name of the flat test dataset.')
    parser.add_argument('--output', type=str, help='Optional JSON output path. Defaults to data/flat_evaluation_results.json.')


def run(config: Config, args: argparse.Namespace):
    pp = PathProvider(config)

    model_id = args.model_id[0]
    driver_type, _ = parse_dataset_id(model_id)
    if driver_type != DriverType.POSTGRES:
        raise ValueError('Flat-feature tree models are currently implemented only for PostgreSQL.')

    dataset_id = create_dataset_id(driver_type, args.test_dataset[0])
    dataset = load_flat_dataset(pp.flat_dataset(dataset_id))
    model = load_flat_model(pp.flat_model(model_id))

    x = transform_dataset_features(model.feature_extractor, dataset, dataset_id)
    predicted = model.predict(x)
    actual = dataset.y()
    metrics = compute_metrics(predicted, actual)
    print_metrics('Evaluation metrics', metrics)

    rows = []
    for item, prediction in zip(dataset, predicted):
        rows.append({
            'id': item.query_id,
            'label': item.label,
            'predicted': float(prediction),
            'measured': float(item.latency),
            'absolute_error': abs(float(prediction) - float(item.latency)),
        })

    output = args.output or os.path.join(config.results_directory, 'flat_evaluation_results.json')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as file:
        json.dump({'metrics': metrics, 'results': rows}, file, indent=4, cls=JsonEncoder)
    print(f'\nSaved results to {output}')


if __name__ == '__main__':
    main()
