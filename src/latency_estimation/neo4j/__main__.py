import os
import argparse
import json
from common.utils import JsonEncoder, auto_close, exit_with_error
from common.query_registry import QueryDef
from latency_estimation.common import format_latency, load_queries, parse_queries, truncate_query, prune_dataset
from latency_estimation.config import TrainConfig, TestConfig
from latency_estimation.neo4j.context import Neo4jContext
from latency_estimation.neo4j.plan_structured_network import PlanStructuredNetwork
from latency_estimation.neo4j.trainer import Trainer
from latency_estimation.neo4j.latency_estimator import LatencyEstimator
from latency_estimation.neo4j.model_evaluator import ModelEvaluator
from latency_estimation.neo4j.feature_extractor import FeatureExtractor

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Neo4j QPP-Net')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_args(subparsers.add_parser('train', help='Train a new QPP-Net model'))
    test_args(subparsers.add_parser('test', help='Test a trained QPP-Net model'))
    estimate_args(subparsers.add_parser('estimate', help='Estimate query latency using a trained QPP-Net model'))

    from common.config import DatasetName
    args = parser.parse_args(rawArgs)

    dataset = DatasetName.TPCH if args.command == 'train' else DatasetName.EDBT
    ctx = Neo4jContext.create(dataset=dataset)

    with auto_close(ctx):
        if args.command == 'train':
            train_run(args, ctx)
        elif args.command == 'test':
            test_run(args, ctx)
        elif args.command == 'estimate':
            estimate_run(args, ctx)

def train_args(parser: argparse.ArgumentParser):
    TrainConfig.neo4j().add_arguments(parser)

def train_run(args: argparse.Namespace, ctx: Neo4jContext):
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
    print(f'Validation set (original): {len(bundle.val)} queries')

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
    TestConfig.neo4j().add_arguments(parser)

def test_run(args: argparse.Namespace, ctx: Neo4jContext):
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
    results = evaluator.evaluate_multiple_queries(test_queries, config.num_runs)
    evaluator.print_summary(results)

    # Save results
    results_path = os.path.join(ctx.config.results_directory, 'evaluation_results.json')
    print(f'\nSaving results to {results_path}...')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4, cls=JsonEncoder)

def estimate_args(parser: argparse.ArgumentParser):
    # Query input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--query', '-q', type=str, help='Single Cypher query to estimate')
    input_group.add_argument('--file', '-f', type=str, help='File containing queries (one per line or semicolon-separated)')

    parser.add_argument('--checkpoint', '-c', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output including query plans')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--quiet', action='store_true', help='Only output the estimated latency value(s)')

def estimate_run(args: argparse.Namespace, ctx: Neo4jContext):
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
                print(f'  Root operator: {plan.get("operatorType", "Unknown")}')
                print(f'  Estimated rows: {plan.get("args", {}).get("EstimatedRows", "N/A")}')

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
