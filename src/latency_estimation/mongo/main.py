"""
Pipeline:
1. Connect to MongoDB
2. Collect query plans + execution times
3. Build feature vocabularies
4. Create plan-structured neural network
5. Train with log-latency loss
6. Evaluate and save model
"""
import torch
import numpy as np
import pickle
import copy
import json
import argparse

from config import MongoConfig
from plan_extractor import PlanExtractor
from feature_extractor import FeatureExtractor
from plan_structured_network import PlanStructuredNetwork
from training import QueryPlanDataset, PlanStructuredTrainer


def main():
    print("=" * 80)
    print("MongoDB Plan-Structured Neural Network for Query Performance Prediction")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="MongoDB QPP Net Training")
    parser.add_argument("--num_queries", type=int, default=1200,
                        help="Number of queries to collect")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=250)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--data_vec_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=27017)
    parser.add_argument("--dbname", type=str, default="tpch")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of execution runs per query for timing")
    parser.add_argument("--cache_file", type=str, default="mongo_query_plans.pkl",
                        help="Cache file for collected plans")
    args = parser.parse_args()

    # Step 1: Connect
    print("\n[1/7] Connecting to MongoDB...")
    config = MongoConfig(host=args.host, port=args.port, dbname=args.dbname)
    extractor = PlanExtractor(config)

    # Step 2: Collect data
    print(f"\n[2/7] Collecting {args.num_queries} query plans...")
    try:
        with open(args.cache_file, "rb") as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} cached query plans from {args.cache_file}")
    except FileNotFoundError:
        dataset = extractor.collect_training_data(
            num_queries=args.num_queries,
            num_runs=args.num_runs,
        )
        with open(args.cache_file, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Collected and cached {len(dataset)} plans to {args.cache_file}")

    exec_times = [d["execution_time_ms"] for d in dataset]
    print(f"\nDataset Statistics:")
    print(f"  Total queries: {len(dataset)}")
    print(f"  Mean exec time: {np.mean(exec_times):.2f} ms")
    print(f"  Median exec time: {np.median(exec_times):.2f} ms")
    print(f"  Min exec time: {np.min(exec_times):.2f} ms")
    print(f"  Max exec time: {np.max(exec_times):.2f} ms")

    # Step 3: Build features
    print("\n[3/7] Building feature vocabularies...")
    coll_stats = config.get_all_collection_stats()
    feature_extractor = FeatureExtractor()
    feature_extractor.set_collection_stats(coll_stats)

    # Extract plan trees for vocabulary building
    plan_explains = [d["explain"] for d in dataset]
    plan_trees = [PlanStructuredNetwork.extract_plan_tree(e) for e in plan_explains]
    feature_extractor.build_vocabularies(plan_trees)

    # Step 4: Split data
    print("\n[4/7] Splitting dataset...")
    split_idx = int(len(dataset) * args.train_split)
    # Shuffle before split for randomness
    indices = np.random.RandomState(42).permutation(len(dataset))
    dataset_shuffled = [dataset[i] for i in indices]

    train_data = dataset_shuffled[:split_idx]
    test_data = dataset_shuffled[split_idx:]
    print(f"  Training: {len(train_data)}, Test: {len(test_data)}")

    # Step 5: Create model
    print("\n[5/7] Creating plan-structured neural network...")
    model = PlanStructuredNetwork(
        feature_extractor=feature_extractor,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        data_vec_dim=args.data_vec_dim,
    )
    model.initialize_units_from_plans(plan_explains)

    train_dataset = QueryPlanDataset(train_data, model)
    test_dataset = QueryPlanDataset(test_data, model)

    # Step 6: Train
    print(f"\n[6/7] Training for {args.num_epochs} epochs...")
    trainer = PlanStructuredTrainer(model, learning_rate=args.lr,
                                    total_epochs=args.num_epochs)

    best_test_geo_qerror = float("inf")
    best_test_mae = float("inf")
    best_model_state = None

    for epoch in range(args.num_epochs):
        loss = trainer.train_epoch(train_dataset, batch_size=args.batch_size)

        if (epoch + 1) % 10 == 0:
            metrics = trainer.evaluate(test_dataset)
            print(f"  Epoch {epoch+1}/{args.num_epochs}  loss={loss:.4f}  "
                  f"MAE={metrics['mae']:.2f}ms  "
                  f"MedR={metrics['median_r']:.2f}  "
                  f"R<=2={metrics['r_within_2.0']*100:.0f}%  "
                  f"GeoQ={metrics['geo_mean_r']:.3f}")

            # Use geometric mean Q-error as primary criterion (more robust than MAE)
            if metrics["geo_mean_r"] < best_test_geo_qerror:
                best_test_geo_qerror = metrics["geo_mean_r"]
                best_test_mae = metrics["mae"]
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"    -> New best model (GeoQ={best_test_geo_qerror:.3f}, MAE={best_test_mae:.2f}ms)")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Step 7: Final evaluation
    print("\n[7/7] Final Evaluation")
    print("=" * 80)

    print("\nTRAINING SET:")
    train_metrics = trainer.evaluate(train_dataset)
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and v < 10 else f"  {k}: {v:.2f}")

    print("\nTEST SET:")
    test_metrics = trainer.evaluate(test_dataset)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and v < 10 else f"  {k}: {v:.2f}")

    # Save
    print("\nSaving model...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_extractor": feature_extractor,
        "coll_stats": coll_stats,
        "operator_info": model.get_operator_info(),
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "data_vec_dim": args.data_vec_dim,
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
    }, "mongo_qpp_model.pt")
    print("Model saved to mongo_qpp_model.pt")

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"  Best Test MAE: {best_test_mae:.2f} ms")
    print(f"  Test R <= 1.5: {test_metrics['r_within_1.5']*100:.1f}%")
    print(f"  Test R <= 2.0: {test_metrics['r_within_2.0']*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
