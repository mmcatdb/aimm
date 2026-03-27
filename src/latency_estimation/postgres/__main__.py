import os
import argparse
import json
from common.utils import JsonEncoder, auto_close, exit_with_error, print_warning
from common.query_registry import QueryDef
from latency_estimation.common import format_latency, load_queries, parse_queries, truncate_query, prune_dataset
from latency_estimation.config import TrainConfig, TestConfig, DatasetConfig
from latency_estimation.postgres.context import PostgresContext
from latency_estimation.postgres.plan_structured_network import PlanStructuredNetwork, TsneItem
from latency_estimation.postgres.trainer import Trainer
from latency_estimation.postgres.latency_estimator import LatencyEstimator
from latency_estimation.postgres.model_evaluator import ModelEvaluator
from latency_estimation.postgres.feature_extractor import FeatureExtractor

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Postgres QPP-Net')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_args(subparsers.add_parser('train', help='Train a new QPP-Net model'))
    test_args(subparsers.add_parser('test', help='Test a trained QPP-Net model'))
    estimate_args(subparsers.add_parser('estimate', help='Estimate query latency using a trained QPP-Net model'))
    tsne_args(subparsers.add_parser('tsne', help='Generate t-SNE data for all plans in the training dataset'))

    from common.config import DatasetName
    args = parser.parse_args(rawArgs)

    dataset = DatasetName.TPCH if args.command == 'train' or args.command == 'tsne' else DatasetName.EDBT
    ctx = PostgresContext.create(dataset=dataset)

    with auto_close(ctx):
        if args.command == 'train':
            train_run(args, ctx)
        elif args.command == 'test':
            test_run(args, ctx)
        elif args.command == 'estimate':
            estimate_run(args, ctx)
        elif args.command == 'tsne':
            tsne_run(args, ctx)

def train_args(parser: argparse.ArgumentParser):
    TrainConfig.postgres().add_arguments(parser)

def train_run(args: argparse.Namespace, ctx: PostgresContext):
    config = TrainConfig.from_arguments(ctx.config, args)

    print(f'\n[1/7] Configuration:')
    print(config)

    print(f'\n[2/7] Collecting {config.dataset.num_queries} query plans...')
    bundle = ctx.load_or_create_dataset(config.dataset)
    combined = bundle.train + bundle.val

    print('\n[3/7] Building feature vocabularies...')
    feature_extractor = FeatureExtractor()
    feature_extractor.build_vocabularies([item.plan for item in combined])

    print(f'Training set: {len(bundle.train)} queries')
    print(f'Validation set: {len(bundle.val)} queries')

    print('\n[5/7] Creating plan-structured neural network...')
    model = PlanStructuredNetwork.from_plans(config.model, ctx.config.device, feature_extractor, [item.plan for item in bundle.train])
    model.print_summary()
    ctx.save_available_operators(model)

    val_dataset = prune_dataset(bundle.val, model)
    print(f'Validation set (pruned): {len(val_dataset)} queries')

    if config.dry_run:
        print('\nDry run completed. Exiting before training.')
        return

    print(f'\n[6/7] Training for {config.num_epochs} epochs...')
    trainer = Trainer(model, config.learning_rate, config.batch_size)

    trainer.train_epochs(bundle.train, val_dataset, config.num_epochs, ctx)

def test_args(parser: argparse.ArgumentParser):
    TestConfig.postgres().add_arguments(parser)

    parser.add_argument('--no-actual', action='store_true', help='Skip actual execution time measurement.')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots.')

def test_run(args: argparse.Namespace, ctx: PostgresContext):
    config = TestConfig.from_arguments(args)

    if config.queries:
        test_queries: list[QueryDef[str]] = []
        for i, content in enumerate(config.queries, 1):
            test_queries.append(QueryDef.create_from_content('custom', i, 1.0, 'Custom Query', content))

        print(f'\nAdded {len(test_queries)} custom query/queries')
    else:
        print('\nGenerating test queries...')
        test_queries = ctx.database().get_query_defs()

    if not test_queries:
        exit_with_error('No queries to test. Provide queries with --query or use the built-in test queries.')

    print(f'Total queries to test: {len(test_queries)}')

    # Run evaluation
    model = ctx.load_model(config.checkpoint)
    evaluator = ModelEvaluator(ctx.extractor, model)
    results = evaluator.evaluate_multiple_queries(
        test_queries,
        measure_actual=not args.no_actual,
        num_runs=config.num_runs
    )
    evaluator.print_summary(results)

    # Save results
    results_path = os.path.join(ctx.config.results_directory, 'evaluation_results.json')
    print(f'\nSaving results to {results_path}...')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4, cls=JsonEncoder)

    # Generate plots
    if not args.no_plots:
        try:
            plot_path = os.path.join(ctx.config.results_directory, 'evaluation_plots.png')
            evaluator.plot_results(results, save_path=plot_path)
        except Exception as e:
            print_warning('Could not generate plots.', e)

def estimate_args(parser: argparse.ArgumentParser):
    # Query input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--query', '-q', type=str, help='Single SQL query to estimate')
    input_group.add_argument('--file', '-f', type=str, help='File containing queries (one per line or semicolon-separated)')

    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output including query plans')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--quiet', action='store_true', help='Only output the estimated latency value(s)')

def estimate_run(args: argparse.Namespace, ctx: PostgresContext):
    queries = load_queries(args, parse_queries)

    model = ctx.load_model(args.checkpoint)

    estimator = LatencyEstimator(ctx.extractor, model)
    results = estimator.estimate_batch(queries)

    # Output results
    if args.json:
        import json
        output = []
        for query, latency, plan in results:
            item: dict = {
                'query': query,
                'estimated_latency_ms': latency,
                'estimated_latency_formatted': format_latency(latency) if latency else None
            }
            if args.verbose:
                item['plan'] = plan
            if 'error' in plan:
                item['error'] = plan['error']
            output.append(item)
        print(json.dumps(output, indent=4))
    elif args.quiet:
        for query, latency, plan in results:
            if latency is not None:
                print(f'{latency:.6f}')
            else:
                print('ERROR')
    else:
        # Standard output
        if len(results) == 1:
            query, latency, plan = results[0]
            if 'error' in plan:
                exit_with_error(plan["error"])

            print(f'Query: {query.strip()}')
            print(f'Estimated latency: {format_latency(latency)}')

            if args.verbose:
                print(f'\nQuery Plan:')
                print(f'  Root operator: {plan.get("Node Type", "Unknown")}')
                print(f'  Estimated rows: {plan.get("Plan Rows", "N/A")}')
                print(f'  Total cost: {plan.get("Total Cost", "N/A")}')

        else:
            # Multiple queries - table format
            print(f'{"#":<4} {"Query":<60} {"Estimated Latency":<20}')
            print('-' * 84)

            for i, (query, latency, plan) in enumerate(results, 1):
                query_display = truncate_query(query)
                if 'error' in plan:
                    latency_str = f'ERROR: {plan["error"][:30]}'
                else:
                    latency_str = format_latency(latency)
                print(f'{i:<4} {query_display:<60} {latency_str:<20}')

            # Summary
            valid_latencies = [lat for _, lat, plan in results if lat is not None and 'error' not in plan]
            if valid_latencies:
                print('-' * 84)
                print(f'Total queries: {len(results)}')
                print(f'Successful estimates: {len(valid_latencies)}')
                print(f'Average estimated latency: {format_latency(sum(valid_latencies) / len(valid_latencies))}')


def tsne_args(parser: argparse.ArgumentParser):
    train = TrainConfig.postgres()
    train.dataset.add_arguments(parser)
    parser.add_argument('--checkpoint', '-c', type=str, help='Path to model checkpoint. Defaults to the "best" model.')

def tsne_run(args: argparse.Namespace, ctx: PostgresContext):
    dataset_config = DatasetConfig.from_arguments(ctx.config, args)

    bundle = ctx.load_or_create_dataset(dataset_config)

    model = ctx.load_model(args.checkpoint)
    val_dataset = prune_dataset(bundle.val, model)

    tsne_items = list[TsneItem]()
    for item in val_dataset:
        tsne_items.extend(model.get_tsne_data(item.plan))

    tsne_by_operator: dict[str, list[TsneItem]] = {}
    for item in tsne_items:
        key = item.operator.key()
        if key not in tsne_by_operator:
            tsne_by_operator[key] = []
        tsne_by_operator[key].append(item)

    for items in tsne_by_operator.values():
        try:
            tsne_for_operator(items)
        except Exception as e:
            print_warning('Could not generate t-SNE plot for operator.', e)

def tsne_for_operator(items: list[TsneItem]):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    operator = items[0].operator

    print(f'\nOperator: {operator.key()}, Items: {len(items)}')
    features = np.array([item.features for item in items])
    estimated = np.array([item.estimated for item in items])
    extracted = np.array([item.extracted for item in items])
    difference = extracted - estimated

    perplexity = min(30, (len(features) - 1) // 3)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    x = StandardScaler().fit_transform(features)

    if np.var(x) < 1e-8:
        # There is no variance in the features, t-SNE will fail. Skip in this case.
        print_warning('No variance in features, skipping t-SNE for this operator.')
        return

    X_2d = tsne.fit_transform(x)

    plt.figure(figsize=(16, 5))
    plt.suptitle(f't-SNE of input features for operator: {operator.key()}')

    plt.subplot(1, 3, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=estimated)
    plt.colorbar()
    plt.title("Estimated latency")

    plt.subplot(1, 3, 2)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=extracted)
    plt.colorbar()
    plt.title("Extracted latency")

    plt.subplot(1, 3, 3)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=difference)
    plt.colorbar()
    plt.title("Difference")

    plt.show()

if __name__ == '__main__':
    main()
