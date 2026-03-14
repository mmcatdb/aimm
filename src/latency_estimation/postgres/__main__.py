import os
import numpy as np
import argparse
import json
from common.utils import JsonEncoder, auto_close, exit_with_error
from common.database import TestQuery
from latency_estimation.common import format_latency, load_queries, parse_queries, print_dataset_summary, truncate_query
from latency_estimation.train_config import TrainConfig
from latency_estimation.postgres.context import PostgresContext
from latency_estimation.postgres.plan_structured_network import PlanStructuredNetwork
from latency_estimation.postgres.trainer import PlanStructuredTrainer
from latency_estimation.postgres.latency_estimator import LatencyEstimator
from latency_estimation.postgres.model_evaluator import ModelEvaluator
from latency_estimation.postgres.feature_extractor import FeatureExtractor

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Postgres QPP-Net')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_args(subparsers.add_parser('train', help='Train a new QPP-Net model'))
    evaluate_args(subparsers.add_parser('evaluate', help='Evaluate a trained QPP-Net model'))
    estimate_args(subparsers.add_parser('estimate', help='Estimate query latency using a trained QPP-Net model'))

    args = parser.parse_args(rawArgs)

    ctx = PostgresContext.create()

    with auto_close(ctx):
        if args.command == 'train':
            train_run(args, ctx)
        elif args.command == 'evaluate':
            evaluate_run(args, ctx)
        elif args.command == 'estimate':
            estimate_run(args, ctx)

def train_args(parser: argparse.ArgumentParser):
    TrainConfig.postgres().add_arguments(parser)

def train_run(args: argparse.Namespace, ctx: PostgresContext):
    config = TrainConfig.from_arguments(args)

    print(f'\n[2/7] Collecting {config.num_queries} query plans...')
    print('This may take a while as each query is executed...')

    dataset = ctx.load_dataset(config.num_queries)

    print(f'\nDataset Statistics:')
    print_dataset_summary(dataset)

    print('\n[3/7] Building feature vocabularies...')
    feature_extractor = FeatureExtractor()
    feature_extractor.build_vocabularies(dataset.plans)

    print('\n[4/7] Splitting dataset...')
    split_index = int(len(dataset) * config.train_split)
    indexes = np.random.permutation(len(dataset))

    train_dataset = dataset.subset(indexes[:split_index])
    test_dataset = dataset.subset(indexes[split_index:])

    print(f'Training set: {len(train_dataset)} queries')
    print(f'Test set: {len(test_dataset)} queries')

    print('\n[5/7] Creating plan-structured neural network...')
    model = PlanStructuredNetwork.from_plans(config.model, feature_extractor, train_dataset.plans)
    model.print_summary()

    print(f'\n[6/7] Training for {config.num_epochs} epochs...')
    trainer = PlanStructuredTrainer(model, config.learning_rate, config.batch_size)

    best_test_mae = float('inf')

    for epoch in range(config.num_epochs):
        loss = trainer.train_epoch(train_dataset)

        if (epoch + 1) % 5 == 0:
            print(f'\nEpoch {epoch + 1}/{config.num_epochs}')
            print(f'  Training Loss: {loss:.4f}')

            # Evaluate on test set
            metrics = trainer.evaluate(test_dataset)
            print(f'  Test MAE: {metrics["mae"]:.2f} ms')
            print(f'  Test Relative Error: {metrics["mre"]:.4f}')
            print(f'  Test R ≤ 1.5: {metrics["le1.5_q_error"] * 100:.1f}%')
            print(f'  Test R ≤ 2.0: {metrics["le2.0_q_error"] * 100:.1f}%')

            # Track and save best model
            if metrics['mae'] < best_test_mae:
                best_test_mae = metrics['mae']
                ctx.save_checkpoint('best', model, trainer, metrics)
                print(f'  ✓ New best model!')

    # Step 7: Final evaluation
    print('\n[7/7] Final Evaluation...')
    print('\n' + '=' * 80)
    print('TRAINING SET PERFORMANCE')
    print('=' * 80)
    train_metrics = trainer.evaluate(train_dataset)
    for metric, value in train_metrics.items():
        if 'error' in metric or 'mae' in metric:
            print(f'  {metric}: {value:.2f}')
        else:
            print(f'  {metric}: {value:.4f}')

    print('\n' + '=' * 80)
    print('TEST SET PERFORMANCE')
    print('=' * 80)
    test_metrics = trainer.evaluate(test_dataset)
    for metric, value in test_metrics.items():
        if 'error' in metric or 'mae' in metric:
            print(f'  {metric}: {value:.2f}')
        else:
            print(f'  {metric}: {value:.4f}')

    # Save model with operator information
    print('\n' + '=' * 80)
    print('Saving model...')
    print('=' * 80)

    ctx.save_checkpoint('final', model, trainer, test_metrics)

    print('\n' + '=' * 80)
    print('Training complete!')
    print('=' * 80)
    print(f'\nFinal Test MAE: {test_metrics["mae"]:.2f} ms')
    print(f'Final Test Relative Error: {test_metrics["mre"]:.4f}')
    print(f'Estimations within factor of 1.5: {test_metrics["le1.5_q_error"] * 100:.1f}%')

def evaluate_args(parser: argparse.ArgumentParser):
    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to trained model')
    parser.add_argument('--no-actual', action='store_true', help='Skip actual execution time measurement')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs for actual execution measurement')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--query', '-q', type=str, action='append', dest='queries', help='Additional SQL query to evaluate (can be used multiple times)')
    parser.add_argument('--query-only', '-qo', action='store_true', help='Only evaluate the provided --query arguments, skip built-in test queries')

def evaluate_run(args: argparse.Namespace, ctx: PostgresContext):
    model = ctx.load_model(args.checkpoint)

    print('\nGenerating test queries...')
    test_queries: list[TestQuery] = [] if args.query_only else ctx.database.get_test_queries()

    # Add user-provided queries
    if args.queries:
        for i, content in enumerate(args.queries, 1):
            query = TestQuery(f'custom-{i}', f'Custom Query {i}', content)
            test_queries.append(query)
        print(f'Added {len(args.queries)} custom query/queries')

    if not test_queries:
        print('Error: No queries to evaluate. Provide queries with --query or remove --query-only flag.')
        return

    print(f'Total queries to evaluate: {len(test_queries)}')

    # Run evaluation
    evaluator = ModelEvaluator(ctx.extractor, model)
    results = evaluator.evaluate_multiple_queries(
        test_queries,
        measure_actual=not args.no_actual,
        num_runs=args.runs
    )
    evaluator.print_summary(results)

    results_path = os.path.join(ctx.config.results_directory, 'evaluation_results.json')
    plot_path = os.path.join(ctx.config.results_directory, 'evaluation_plots.png')

    # Save results
    print(f'\nSaving results to {results_path}...')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=2, cls=JsonEncoder)

    # Generate plots
    if not args.no_plots:
        try:
            evaluator.plot_results(results, save_path=plot_path)
        except Exception as e:
            print(f'Error: Could not generate plots: {e}')

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
                'estimated_latency_seconds': latency,
                'estimated_latency_formatted': format_latency(latency) if latency else None
            }
            if args.verbose:
                item['plan'] = plan
            if 'error' in plan:
                item['error'] = plan['error']
            output.append(item)
        print(json.dumps(output, indent=2))
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

if __name__ == '__main__':
    main()
