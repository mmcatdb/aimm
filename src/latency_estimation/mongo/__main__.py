import os
from typing import Any
import numpy as np
import argparse
import json
from common.utils import JsonEncoder, auto_close, exit_with_error
from common.database import TestQuery, MongoQuery, try_parse_mongo_query
from common.drivers import MongoDriver
from latency_estimation.common import print_dataset_summary
from latency_estimation.config import TrainConfig, TestConfig
from latency_estimation.mongo.context import MongoContext
from latency_estimation.mongo.plan_structured_network import PlanStructuredNetwork
from latency_estimation.mongo.trainer import PlanStructuredTrainer
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
    config = TrainConfig.from_arguments(args)

    print(f'\n[1/7] Configuration:')
    print(config)

    print(f'\n[2/7] Collecting {config.num_queries} query plans...')
    dataset = ctx.load_or_create_dataset(config.num_queries, config.num_runs)

    print(f'\nDataset Statistics:')
    print_dataset_summary(dataset)

    print('\n[3/7] Building feature vocabularies...')
    feature_extractor = FeatureExtractor()
    feature_extractor.build_collection_stats(ctx.driver)
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
    ctx.save_available_operators(model)

    if config.dry_run:
        print('\nDry run complete. Exiting before training.')
        return

    print(f'\n[6/7] Training for {config.num_epochs} epochs...')
    trainer = PlanStructuredTrainer(model, config.learning_rate, config.batch_size, config.num_epochs, args.warmup_epochs)

    best_test_geo_qerror = float('inf')
    best_test_mae = float('inf')

    for epoch in range(config.num_epochs):
        loss = trainer.train_epoch(train_dataset)

        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch {epoch + 1}/{config.num_epochs}')
            print(f'  Training Loss: {loss:.4f}')

            # Evaluate on test set
            metrics = trainer.evaluate(test_dataset)
            print(f'  MAE={metrics["mae"]:.2f}ms')
            print(f'  MedR={metrics["median_r"]:.2f}')
            print(f'  R<=2={metrics["r_within_2.0"] * 100:.0f}%')
            print(f'  GeoQ={metrics["geo_mean_r"]:.3f}')

            # Track and save best model
            # Use geometric mean Q-error as primary criterion (more robust than MAE)
            if metrics['geo_mean_r'] < best_test_geo_qerror:
                best_test_geo_qerror = metrics['geo_mean_r']
                best_test_mae = metrics['mae']
                ctx.save_checkpoint('best', model, trainer, metrics)
                print(f'    -> New best model (GeoQ={best_test_geo_qerror:.3f}, MAE={best_test_mae:.2f}ms)')

    # Step 7: Final evaluation
    print('\n[7/7] Final Evaluation...')
    print('\n' + '=' * 80)

    print('TRAINING SET PERFORMANCE')
    print('=' * 80)

    train_metrics = trainer.evaluate(train_dataset)
    for k, v in train_metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) and v < 10 else f'  {k}: {v:.2f}')

    print('\n' + '=' * 80)
    print('TEST SET PERFORMANCE')
    print('=' * 80)
    test_metrics = trainer.evaluate(test_dataset)
    for k, v in test_metrics.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) and v < 10 else f'  {k}: {v:.2f}')

    print('\n' + '=' * 80)
    print('Saving model...')
    print('=' * 80)

    ctx.save_checkpoint('final', model, trainer, test_metrics)

    print('\n' + '=' * 80)
    print('Training complete!')
    print(f'  Best Test MAE: {best_test_mae:.2f} ms')
    print(f'  Test R <= 1.5: {test_metrics["r_within_1.5"]*100:.1f}%')
    print(f'  Test R <= 2.0: {test_metrics["r_within_2.0"]*100:.1f}%')
    print('=' * 80)

def get_all_collection_stats(driver: MongoDriver) -> dict[str, dict[str, Any]]:
    """Get statistics for all collections."""
    db = driver.database()
    stats = {}
    for name in db.list_collection_names():
        if not name.startswith("system."):
            stats[name] = _get_collection_stats(driver, name)
    return stats

def _get_collection_stats(driver: MongoDriver, collection_name: str) -> dict[str, Any]:
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
        test_queries: list[TestQuery[MongoQuery]] = []
        for i, content in enumerate(config.queries, 1):
            mongo_query = try_parse_mongo_query(content)
            if mongo_query is not None:
                test_queries.append(TestQuery(f'custom-{i}', f'Custom Query {i}', mongo_query))

        print(f'\nAdded {len(test_queries)} custom query/queries')
    else:
        print('\nGenerating test queries...')
        test_queries = ctx.database.get_test_queries()

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
