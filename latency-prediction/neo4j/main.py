import torch
import numpy as np
import pickle
import argparse
import time

from plan_extractor import PlanExtractor
from feature_extractor import FeatureExtractor
from plan_structured_network import PlanStructuredNetwork
from training import (
    Neo4jQueryPlanDataset,
    PlanStructuredTrainer,
    train_model
)

def save_workload_data(
    queries: list[str],
    plans: list[dict],
    execution_times: list[float],
    filename: str = 'data/neo4j_workload_data.pkl'
):
    """
    Save collected workload data to disk.

    Args:
        queries: List of query strings
        plans: List of query plans
        execution_times: List of execution times
        filename: Path to save file
    """
    data = {
        'queries': queries,
        'plans': plans,
        'execution_times': execution_times
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nWorkload data saved to {filename}")


def load_workload_data(filename: str = 'data/neo4j_workload_data.pkl') -> tuple[list[str], list[dict], list[float]]:
    """
    Load previously collected workload data.

    Args:
        filename: Path to saved data file

    Returns:
        Tuple of (queries, plans, execution_times)
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"\nWorkload data loaded from {filename}")
    return data['queries'], data['plans'], data['execution_times']


def main():
    parser = argparse.ArgumentParser(description='Train Neo4j Query Performance Predictor')
    parser.add_argument('--num-queries', type=int, default=250, help='Number of query variants to generate')
    parser.add_argument('--num-runs', type=int, default=1, help='Number of executions per query for averaging')
    parser.add_argument('--num-epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for neural units')
    parser.add_argument('--num-layers', type=int, default=5, help='Number of layers per neural unit')
    parser.add_argument('--data-vec-dim', type=int, default=32, help='Data vector dimension')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--load-workload', type=str, default=None, help='Path to load previously collected workload data')
    parser.add_argument('--save-workload', type=str, default='data/neo4j_workload_data.pkl', help='Path to save collected workload data')
    parser.add_argument('--checkpoint-path', type=str, default='data/neo4j_qpp_checkpoint.pt', help='Path to save model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for training')

    args = parser.parse_args()

    print(f"\nConfiguration:")
    print(f"  Number of queries: {args.num_queries}")
    print(f"  Runs per query: {args.num_runs}")
    print(f"  Training epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Data vector dimension: {args.data_vec_dim}")
    print(f"  Validation split: {args.val_split}")
    print(f"  Device: {args.device}")
    print()


    if args.load_workload:
        print(f"Loading workload data from {args.load_workload}...")
        queries, plans, execution_times = load_workload_data(args.load_workload)
    else:
        print("Connecting to Neo4j and collecting workload...")
        extractor = PlanExtractor()

        try:
            queries, plans, execution_times = extractor.collect_workload(
                num_queries=args.num_queries,
                num_runs_per_query=args.num_runs
            )

            # Save workload data for future use
            save_workload_data(queries, plans, execution_times, args.save_workload)

        finally:
            extractor.close()

    print(f"\nCollected {len(queries)} queries with execution times")


    feature_extractor = FeatureExtractor()
    feature_extractor.build_vocabularies(plans)

    # Get feature dimension
    sample_features = feature_extractor.extract_features(plans[0])
    feature_dim = len(sample_features)
    print(f"\nFeature vector dimension: {feature_dim}")


    num_val = int(len(queries) * args.val_split)
    num_train = len(queries) - num_val

    # Shuffle data
    indices = np.random.permutation(len(queries))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_queries = [queries[i] for i in train_indices]
    train_plans = [plans[i] for i in train_indices]
    train_times = [execution_times[i] for i in train_indices]

    val_queries = [queries[i] for i in val_indices]
    val_plans = [plans[i] for i in val_indices]
    val_times = [execution_times[i] for i in val_indices]

    print(f"\nTraining set: {num_train} queries")
    print(f"Validation set: {num_val} queries")

    # Create datasets
    train_dataset = Neo4jQueryPlanDataset(train_queries, train_plans, train_times)
    val_dataset = Neo4jQueryPlanDataset(val_queries, val_plans, val_times) if num_val > 0 else None

    print("\n" + "=" * 70)

    start_time = time.time()

    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        feature_extractor=feature_extractor,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        data_vec_dim=args.data_vec_dim,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.2f} seconds")

    print("\n" + "=" * 70)
    print("Evalutation")

    trainer = PlanStructuredTrainer(model, device=args.device)

    print("\nTraining set performance:")
    train_metrics = trainer.evaluate(train_dataset, batch_size=args.batch_size)
    for metric_name, value in train_metrics.items():
        print(f"  {metric_name}: {value:.6f}")

    if val_dataset:
        print("\nValidation set performance:")
        val_metrics = trainer.evaluate(val_dataset, batch_size=args.batch_size)
        for metric_name, value in val_metrics.items():
            print(f"  {metric_name}: {value:.6f}")

    feature_extractor_path = args.checkpoint_path.replace('.pt', '_feature_extractor.pkl')
    with open(feature_extractor_path, 'wb') as f:
        pickle.dump(feature_extractor, f)
    print(f"\nFeature extractor saved to {feature_extractor_path}")

    print("=" * 70)
    print(f"\nModel checkpoint: {args.checkpoint_path}")
    print(f"Feature extractor: {feature_extractor_path}")
    print(f"Total time: {training_time:.2f} seconds")

if __name__ == '__main__':
    main()
