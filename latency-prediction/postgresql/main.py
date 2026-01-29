"""
Main training and evaluation script.

Orchestrates the entire pipeline:
1. Connect to database
2. Collect query plans
3. Build feature vocabularies
4. Create and train model
5. Evaluate performance
"""
import torch
import numpy as np
import pickle
import copy
import json
import argparse
from config import DatabaseConfig
from plan_extractor import PlanExtractor
from feature_extractor import FeatureExtractor
from plan_structured_network import PlanStructuredNetwork
from training import QueryPlanDataset, PlanStructuredTrainer

def main():
    print("=" * 80)
    print("Plan-Structured Neural Network for Query Performance Prediction")
    print("=" * 80)
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="QPP Network Training")
    parser.add_argument('--num_queries', type=int, default=500, help='Number of queries to collect')
    parser.add_argument('--train_split', type=float, default=0.8, help='Fraction of data for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of hidden layers per neural unit')
    parser.add_argument('--data_vec_dim', type=int, default=32, help='Data vector dimension size')
    
    args = parser.parse_args()
    
    # Configuration
    NUM_QUERIES = args.num_queries
    TRAIN_SPLIT = args.train_split
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    HIDDEN_DIM = args.hidden_dim
    NUM_LAYERS = args.num_layers
    DATA_VEC_DIM = args.data_vec_dim
    
    # Step 1: Initialize database connection
    print("\n[1/7] Connecting to database...")
    config = DatabaseConfig()
    extractor = PlanExtractor(config)
    
    # Step 2: Collect query plans
    print(f"\n[2/7] Collecting {NUM_QUERIES} query plans...")
    print("This may take a while as each query is executed...")
    
    # Try to load cached data first
    try:
        with open('query_plans.pkl', 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} cached query plans")
    except FileNotFoundError:
        dataset = extractor.collect_training_data(NUM_QUERIES)
        # Cache for future runs
        with open('query_plans.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Collected and cached {len(dataset)} query plans")
    
    # Extract plans and execution times
    plans = [item['plan'] for item in dataset]
    execution_times = [item['execution_time'] for item in dataset]
    
    print(f"\nDataset Statistics:")
    print(f"  Total queries: {len(plans)}")
    print(f"  Average execution time: {np.mean(execution_times):.2f} ms")
    print(f"  Min execution time: {np.min(execution_times):.2f} ms")
    print(f"  Max execution time: {np.max(execution_times):.2f} ms")
    print(f"  Median execution time: {np.median(execution_times):.2f} ms")
    
    # Step 3: Build feature extractor
    print("\n[3/7] Building feature vocabularies...")
    feature_extractor = FeatureExtractor()
    feature_extractor.build_vocabularies(plans)
    
    # Step 4: Split into train/test
    print("\n[4/7] Splitting dataset...")
    split_idx = int(len(plans) * TRAIN_SPLIT)
    
    train_plans = plans[:split_idx]
    train_times = execution_times[:split_idx]
    test_plans = plans[split_idx:]
    test_times = execution_times[split_idx:]
    
    print(f"Training set: {len(train_plans)} queries")
    print(f"Test set: {len(test_plans)} queries")
    
    # Create datasets
    train_dataset = QueryPlanDataset(train_plans, train_times)
    test_dataset = QueryPlanDataset(test_plans, test_times)
    
    # Step 5: Create model
    print("\n[5/7] Creating plan-structured neural network...")
    model = PlanStructuredNetwork(
        feature_extractor=feature_extractor,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        data_vec_dim=DATA_VEC_DIM
    )
    
    # Initialize neural units from training data
    print("\nInitializing neural units from training data...")
    model.initialize_units_from_plans(plans)
    
    # Step 6: Train model
    print(f"\n[6/7] Training for {NUM_EPOCHS} epochs...")
    trainer = PlanStructuredTrainer(model, learning_rate=0.001, momentum=0.9)
    
    best_test_mae = float('inf')
    best_model_state = None 

    for epoch in range(NUM_EPOCHS):
        loss = trainer.train_epoch(train_dataset, batch_size=BATCH_SIZE)
        
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"  Training Loss: {loss:.4f}")
            
            # Evaluate on test set
            metrics = trainer.evaluate(test_dataset)
            print(f"  Test MAE: {metrics['mae']:.2f} ms")
            print(f"  Test Relative Error: {metrics['relative_error']:.4f}")
            print(f"  Test R ≤ 1.5: {metrics['r_within_1.5']*100:.1f}%")
            print(f"  Test R ≤ 2.0: {metrics['r_within_2.0']*100:.1f}%")
            
            # Track and save best model
            if metrics['mae'] < best_test_mae:
                best_test_mae = metrics['mae']
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  ✓ New best model!")
    
    # Step 7: Final evaluation
    print("\n[7/7] Final Evaluation...")
    print("\n" + "=" * 80)
    print("TRAINING SET PERFORMANCE")
    print("=" * 80)
    train_metrics = trainer.evaluate(train_dataset)
    for metric, value in train_metrics.items():
        if 'error' in metric or 'mae' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("TEST SET PERFORMANCE")
    print("=" * 80)
    test_metrics = trainer.evaluate(test_dataset)
    for metric, value in test_metrics.items():
        if 'error' in metric or 'mae' in metric:
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value:.4f}")
    
    # Save model with operator information
    print("\n" + "=" * 80)
    print("Saving model...")
    print("=" * 80)
    
    # Extract operator info for reconstruction
    operator_info = model.get_operator_info()
    
    torch.save({
        'model_state_dict': best_model_state,
        'feature_extractor': feature_extractor,
        'operator_info': operator_info,  # Save for model reconstruction
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'data_vec_dim': DATA_VEC_DIM
        },
        'metrics': {
            'train': train_metrics,
            'test': test_metrics
        }
    }, 'qpp_net_model.pt')
    print("Model saved to 'qpp_net_model.pt'")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nFinal Test MAE: {test_metrics['mae']:.2f} ms")
    print(f"Final Test Relative Error: {test_metrics['relative_error']:.4f}")
    print(f"Predictions within factor of 1.5: {test_metrics['r_within_1.5']*100:.1f}%")

if __name__ == '__main__':
    main()
