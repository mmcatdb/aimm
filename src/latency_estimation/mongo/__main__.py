import os
import argparse
import json
from common.utils import JsonEncoder, auto_close, exit_with_error
from common.query_registry import QueryDef
from common.database import MongoQuery, try_parse_mongo_query
from common.drivers import MongoDriver
from latency_estimation.common import prune_dataset
from latency_estimation.config import TrainConfig, TestConfig
from latency_estimation.mongo.context import MongoContext
from latency_estimation.mongo.plan_structured_network import PlanStructuredNetwork
from latency_estimation.mongo.trainer import Trainer
from latency_estimation.mongo.model_evaluator import ModelEvaluator
from latency_estimation.mongo.feature_extractor import FeatureExtractor

def main(rawArgs: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Mongo QPP-Net')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_args(subparsers.add_parser('train', help='Train a new QPP-Net model'))
    test_args(subparsers.add_parser('test', help='Test a trained QPP-Net model'))
    # estimate_args(subparsers.add_parser('estimate', help='Estimate query latency using a trained QPP-Net model'))

    args = parser.parse_args(rawArgs)

    ctx = MongoContext.create()

    with auto_close(ctx):
        if args.command == 'train':
            train_run(args, ctx)
        elif args.command == 'test':
            test_run(args, ctx)
        # elif args.command == 'estimate':
        #     estimate_run(args, ctx)

def train_args(parser: argparse.ArgumentParser):
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Number of warmup epochs with higher learning rate')

    TrainConfig.mongo().add_arguments(parser)

def train_run(args: argparse.Namespace, ctx: MongoContext):
    config = TrainConfig.from_arguments(ctx.config, args)

    print(f'\n[1/7] Configuration:')
    print(config)

    print(f'\n[2/7] Collecting {config.num_queries} query plans...')
    bundle = ctx.load_or_create_dataset(config)
    combined = bundle.train + bundle.val

    print('\n[3/7] Building feature vocabularies...')
    feature_extractor = FeatureExtractor()
    feature_extractor.build_collection_stats(ctx.driver)
    feature_extractor.build_vocabularies([item.plan for item in combined])

    print(f'Training set: {len(bundle.train)} queries')
    print(f'Validation set (original): {len(bundle.val)} queries')

    print('\n[5/7] Creating plan-structured neural network...')
    model = PlanStructuredNetwork.from_plans(config.model, feature_extractor, [item.plan for item in bundle.train])
    model.print_summary()
    ctx.save_available_operators(model)

    val_dataset = prune_dataset(bundle.val, model)
    print(f'Validation set (pruned): {len(val_dataset)} queries')

    if config.dry_run:
        print('\nDry run completed. Exiting before training.')
        return

    print(f'\n[6/7] Training for {config.num_epochs} epochs...')
    trainer = Trainer(model, config.learning_rate, config.batch_size, config.num_epochs, args.warmup_epochs)

    trainer.train_epochs(bundle.train, val_dataset, config.num_epochs, lambda name, metrics: ctx.save_checkpoint(name, model, trainer, metrics))

def get_all_collection_stats(driver: MongoDriver) -> dict[str, dict]:
    """Get statistics for all collections."""
    db = driver.database()
    stats = {}
    for name in db.list_collection_names():
        if not name.startswith("system."):
            stats[name] = _get_collection_stats(driver, name)
    return stats

def _get_collection_stats(driver: MongoDriver, collection_name: str) -> dict:
    """Get collection statistics via collStats command."""
    db = driver.database()
    stats = db.command("collStats", collection_name)
    return {
        'count': stats.get('count', 0),
        'size': stats.get('size', 0),
        'avgObjSize': stats.get('avgObjSize', 0),
        'storageSize': stats.get('storageSize', 0),
        'nindexes': stats.get('nindexes', 0),
        'totalIndexSize': stats.get('totalIndexSize', 0),
    }

def test_args(parser: argparse.ArgumentParser):
    TestConfig.mongo().add_arguments(parser)

def test_run(args: argparse.Namespace, ctx: MongoContext):
    config = TestConfig.from_arguments(args)

    if config.queries:
        test_queries: list[QueryDef[MongoQuery]] = []
        for i, content in enumerate(config.queries, 1):
            mongo_query = try_parse_mongo_query(content)
            if mongo_query is not None:
                test_queries.append(QueryDef.create_from_content('custom', i, 1.0, 'Custom Query', mongo_query))

        print(f'\nAdded {len(test_queries)} custom query/queries')
    else:
        print('\nGenerating test queries...')
        test_queries = ctx.database().get_query_defs()

    if not test_queries:
        exit_with_error('No queries to test. Provide queries with --query or use the built-in test queries.')

    print(f'Total queries to test: {len(test_queries)}')

    model = ctx.load_model(config.checkpoint)
    evaluator = ModelEvaluator(ctx.extractor, model)
    results = evaluator.evaluate_multiple_queries(test_queries, num_runs=config.num_runs)
    evaluator.print_summary(results)

    # Save results
    results_path = os.path.join(ctx.config.results_directory, 'evaluation_results.json')
    print(f'\nSaving results to {results_path}...')
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4, cls=JsonEncoder)

# def estimate_args(parser: argparse.ArgumentParser):
#     pass

# def estimate_run(args: argparse.Namespace, ctx: MongoContext):
#     pass

if __name__ == '__main__':
    main()
