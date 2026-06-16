import argparse
import json
import os
from collections import defaultdict
from statistics import median

from core.config import Config
from core.drivers import DriverType
from core.query import parse_query_instance_id
from core.utils import exit_with_exception
from core.files import JsonEncoder, open_output
from latency_estimation.dataset import create_dataset_id, parse_dataset_id
from latency_estimation.mongo.flat_dataset import load_mongo_flat_dataset
from latency_estimation.mongo.flat_model import compute_metrics, load_flat_model, print_metrics, transform_dataset_features
from providers.path_provider import PathProvider
from tabulate import tabulate


def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Evaluate a MongoDB flat-feature tree latency model on a flat dataset.')
    add_args(parser)
    args = parser.parse_args(rawArgs)

    try:
        run(Config.load(), args)
    except Exception as e:
        exit_with_exception(e)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('model_id', nargs=1, help='Id of the model. Pattern: mongo/{model_name}.')
    parser.add_argument('test_dataset', nargs=1, help='Name of the flat test dataset.')
    parser.add_argument('--output', type=str, help='Optional JSON output path. Defaults to data/mongo_flat_evaluation_results.json.')
    parser.add_argument('--verbose', action='store_true', help='Print worst predictions and per-template error summaries.')
    parser.add_argument('--top-k', type=int, default=20, help='Number of worst individual predictions to print with --verbose.')


def run(config: Config, args: argparse.Namespace):
    pp = PathProvider(config)

    model_id = args.model_id[0]
    driver_type, _ = parse_dataset_id(model_id)
    if driver_type != DriverType.MONGO:
        raise ValueError('MongoDB flat-feature tree evaluation requires a model id with pattern mongo/{model_name}.')

    dataset_id = create_dataset_id(driver_type, args.test_dataset[0])
    dataset = load_mongo_flat_dataset(pp.flat_dataset(dataset_id))
    model = load_flat_model(pp.flat_model(model_id))

    x = transform_dataset_features(model.feature_extractor, dataset, dataset_id)
    predicted = model.predict(x)
    actual = dataset.y()
    metrics = compute_metrics(predicted, actual)
    print_metrics('Evaluation metrics', metrics)

    rows = []
    for item, prediction in zip(dataset, predicted):
        q_value = max(
            float(prediction) / (float(item.latency) + 1e-9),
            float(item.latency) / (float(prediction) + 1e-9),
        )
        rows.append({
            'id': item.query_id,
            'label': item.label,
            'predicted': float(prediction),
            'measured': float(item.latency),
            'absolute_error': abs(float(prediction) - float(item.latency)),
            'q_value': q_value,
        })

    if args.verbose:
        print_verbose_results(rows, args.top_k)

    output = args.output or os.path.join(config.results_directory, 'mongo_flat_evaluation_results.json')
    with open_output(output) as file:
        json.dump({'metrics': metrics, 'results': rows}, file, indent=4, cls=JsonEncoder)
    print(f'\nSaved results to {output}')


def print_verbose_results(rows: list[dict], top_k: int):
    grouped = defaultdict(list)
    for row in rows:
        _, template_name, _ = parse_query_instance_id(row['id'])
        grouped[template_name].append(row)

    template_rows = []
    for template_name, items in grouped.items():
        q_values = [item['q_value'] for item in items]
        signed_relative_errors = [
            (item['predicted'] - item['measured']) / (item['measured'] + 1e-9)
            for item in items
        ]
        template_rows.append({
            'template': template_name,
            'count': len(items),
            'median_q': median(q_values),
            'mean_q': sum(q_values) / len(q_values),
            'within_2': sum(q <= 2.0 for q in q_values) / len(q_values),
            'median_signed_relative_error': median(signed_relative_errors),
            'median_measured': median(item['measured'] for item in items),
            'median_predicted': median(item['predicted'] for item in items),
        })

    template_table = [
        [
            row['template'],
            row['count'],
            row['median_q'],
            row['mean_q'],
            row['within_2'] * 100.0,
            row['median_measured'],
            row['median_predicted'],
            row['median_signed_relative_error'],
        ]
        for row in sorted(template_rows, key=lambda item: item['median_q'], reverse=True)
    ]
    print('\nWorst templates by median R-value:')
    print(tabulate(
        template_table,
        headers=[
            'Template',
            'N',
            'Med R',
            'Mean R',
            'R<=2 %',
            'Med Actual ms',
            'Med Pred ms',
            'Med Signed Rel',
        ],
        floatfmt=('', 'd', '.2f', '.2f', '.1f', '.2f', '.2f', '.2f'),
        tablefmt='github',
    ))

    worst_rows = []
    for row in sorted(rows, key=lambda item: item['q_value'], reverse=True)[:top_k]:
        _, template_name, instance_index = parse_query_instance_id(row['id'])
        signed_relative_error = (row['predicted'] - row['measured']) / (row['measured'] + 1e-9)
        worst_rows.append([
            row['q_value'],
            row['absolute_error'],
            row['predicted'],
            row['measured'],
            signed_relative_error,
            f'{template_name}:{instance_index}',
            _compact_label(row['label']),
        ])

    print(f'\nWorst {min(top_k, len(rows))} predictions by R-value:')
    print(tabulate(
        worst_rows,
        headers=[
            'R',
            'Abs ms',
            'Pred ms',
            'Actual ms',
            'Signed Rel',
            'Query',
            'Title',
        ],
        floatfmt=('.2f', '.2f', '.2f', '.2f', '.2f', '', ''),
        tablefmt='github',
    ))


def _compact_label(label: str, max_length: int = 72) -> str:
    title = label.split(' - ', 1)[1] if ' - ' in label else label
    if len(title) <= max_length:
        return title
    return title[:max_length - 3] + '...'


if __name__ == '__main__':
    main()
