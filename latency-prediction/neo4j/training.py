import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import time
from neo4j import GraphDatabase
import yaml
from plan_structured_network import PlanStructuredNetwork
from feature_extractor import FeatureExtractor


class Neo4jQueryPlanDataset(Dataset):
    """
    Dataset of Neo4j query plans with execution times.
    
    Each item contains:
    - query: The Cypher query string
    - plan: The query execution plan (from EXPLAIN)
    - execution_time: Actual measured execution time in seconds
    """
    
    def __init__(self, queries: List[str], plans: List[Dict], 
                 execution_times: List[float]):
        """
        Args:
            queries: List of Cypher query strings
            plans: List of query execution plans (from EXPLAIN)
            execution_times: List of actual execution times in seconds
        """
        assert len(queries) == len(plans) == len(execution_times), \
            "Queries, plans, and execution times must have same length"
        
        self.queries = queries
        self.plans = plans
        self.execution_times = execution_times
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return {
            'query': self.queries[idx],
            'plan': self.plans[idx],
            'execution_time': self.execution_times[idx]
        }


def compute_plan_structure_hash(plan: Dict) -> str:
    """
    Compute a hash representing the structure of a query plan.
    Plans with identical structure can be batched together.
    
    Args:
        plan: Neo4j query plan (root node)
        
    Returns:
        Hash string representing the plan structure
    """
    def structure_sig(node):
        op_type = node.get('operatorType', 'Unknown').replace('@neo4j', '')
        children = node.get('children', [])
        
        if not children:
            return op_type
        
        # Sort children signatures for consistency
        child_sigs = sorted([structure_sig(child) for child in children])
        return f"{op_type}({','.join(child_sigs)})"
    
    return structure_sig(plan)


def group_plans_by_structure(batch: List[Dict]) -> Dict[str, List[int]]:
    """
    Group plans in a batch by their structure.
    Plans with identical structure can share computation.
    
    Args:
        batch: List of batch items (dicts with 'plan' key)
        
    Returns:
        Mapping from structure hash to indices in batch
    """
    groups = defaultdict(list)
    
    for idx, item in enumerate(batch):
        plan = item['plan']
        structure = compute_plan_structure_hash(plan)
        groups[structure].append(idx)
    
    return groups


class Neo4jConnection:
    """
    Manages connection to Neo4j database for query execution.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Path to YAML config file with Neo4j credentials
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        neo4j_config = config['neo4j']
        self.driver = GraphDatabase.driver(
            neo4j_config['uri'],
            auth=(neo4j_config['user'], neo4j_config['password'])
        )
    
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
    
    def get_plan(self, query: str) -> Dict:
        """
        Get query execution plan using EXPLAIN (no execution).
        
        Args:
            query: Cypher query string
            
        Returns:
            Query plan as dictionary
        """
        with self.driver.session() as session:
            result = session.run(f"EXPLAIN {query}")
            plan = result.consume().plan
            # Neo4j driver already returns plan as dict
            return plan
    
    def get_plan_and_execute(self, query: str, num_runs: int = 3) -> Tuple[Dict, float]:
        """
        Get query plan with EXPLAIN and measure actual execution time.
        
        Note: We use EXPLAIN for the plan (not PROFILE) since PROFILE doesn't
        return the plan structure in the same way.
        
        Args:
            query: Cypher query string
            num_runs: Number of times to execute for averaging
            
        Returns:
            Tuple of (plan, average_execution_time_seconds)
        """
        with self.driver.session() as session:
            # Get plan with EXPLAIN (doesn't execute)
            result = session.run(f"EXPLAIN {query}")
            summary = result.consume()
            plan = summary.plan  # Already a dict
            
            # Measure actual execution time
            execution_times = []
            for _ in range(num_runs):
                start_time = time.time()
                result = session.run(query)
                result.consume()  # Ensure full execution
                end_time = time.time()
                execution_times.append(end_time - start_time)
            
            avg_time = np.mean(execution_times)
            return plan, avg_time


class PlanStructuredTrainer:
    """
    Trainer for plan-structured neural networks (Neo4j version).
    Implements optimized training with batching and caching.
    """
    
    def __init__(self, model: PlanStructuredNetwork, 
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 device: str = 'cpu'):
        """
        Args:
            model: PlanStructuredNetwork instance
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization coefficient
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function: MSE for total query latency
        self.criterion = nn.MSELoss()
        
        # Training statistics
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, batch: List[Dict]) -> torch.Tensor:
        """
        Compute MSLE loss for a batch of query plans.
        Loss = (log(predicted + 1) - log(actual + 1))²
        """
        predictions = []
        targets = []
        
        # Group plans by structure for efficiency
        structure_groups = group_plans_by_structure(batch)
        
        for structure_hash, indices in structure_groups.items():
            # Process plans with same structure together
            for idx in indices:
                plan = batch[idx]['plan']
                execution_time = batch[idx]['execution_time']
                
                # Forward pass through model
                predicted_latency = self.model.forward(plan)
                
                predictions.append(predicted_latency)
                targets.append(torch.tensor([[execution_time]], 
                                           dtype=torch.float32, 
                                           device=self.device))
        
        # Stack predictions and targets
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        # Apply Log transformation before MSE
        log_preds = torch.log1p(torch.abs(predictions)) 
        log_targets = torch.log1p(targets)
        
        # Compute MSE on log values -> MSLE
        loss = self.criterion(log_preds, log_targets)
        
        return loss
    
    def train_batch(self, batch: List[Dict]) -> float:
        """
        Train on a single batch.
        
        Args:
            batch: List of batch items
            
        Returns:
            Loss value for this batch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute loss
        loss = self.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataset: Neo4jQueryPlanDataset, 
                    batch_size: int = 32,
                    shuffle: bool = True) -> float:
        """
        Train for one epoch.
        
        Args:
            dataset: Training dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            
        Returns:
            Average loss for the epoch
        """
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=lambda x: x  # Return list of items as-is
        )
        
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            loss = self.train_batch(batch)
            epoch_losses.append(loss)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Loss: {loss:.6f}")
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, dataset: Neo4jQueryPlanDataset, 
                 batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Evaluation dataset
            batch_size: Batch size
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda x: x
        )
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                for item in batch:
                    plan = item['plan']
                    execution_time = item['execution_time']
                    
                    # Predict
                    predicted_latency = self.model.forward(plan)
                    
                    all_predictions.append(predicted_latency.item())
                    all_targets.append(execution_time)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Compute metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R-value: max(pred/actual, actual/pred)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        r_values = np.maximum(
            (predictions + epsilon) / (targets + epsilon),
            (targets + epsilon) / (predictions + epsilon)
        )
        mean_q_error = np.mean(r_values)
        median_q_error = np.median(r_values)
        p90_q_error = np.percentile(r_values, 90)
        p95_q_error = np.percentile(r_values, 95)
        
        # Relative error
        relative_errors = np.abs(predictions - targets) / (targets + epsilon)
        mean_relative_error = np.mean(relative_errors)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mean_q_error': mean_q_error,
            'median_q_error': median_q_error,
            'p90_q_error': p90_q_error,
            'p95_q_error': p95_q_error,
            'mean_relative_error': mean_relative_error
        }
        
        return metrics
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Current metrics dictionary
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': metrics,
            'operator_info': self.model.get_operator_info()
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded from {path}")
        return checkpoint


def collect_training_data(queries: List[str], 
                         neo4j_conn: Neo4jConnection,
                         num_runs: int = 3) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Collect training data by executing queries and measuring their performance.
    
    Args:
        queries: List of Cypher query strings
        neo4j_conn: Neo4j connection object
        num_runs: Number of executions per query for averaging
        
    Returns:
        Tuple of (queries, plans, execution_times)
    """
    print(f"Collecting training data from {len(queries)} queries...")
    print(f"Each query will be executed {num_runs} times for averaging.")
    
    collected_queries = []
    plans = []
    execution_times = []
    
    for i, query in enumerate(queries):
        try:
            print(f"\nQuery {i+1}/{len(queries)}:")
            print(f"  {query[:100]}..." if len(query) > 100 else f"  {query}")
            
            # Get plan and execution time
            plan, exec_time = neo4j_conn.get_plan_and_execute(query, num_runs)
            
            collected_queries.append(query)
            plans.append(plan)
            execution_times.append(exec_time)
            
            print(f"  Execution time: {exec_time:.4f}s")
            print(f"  Root operator: {plan.get('operatorType', 'Unknown')}")
            
        except Exception as e:
            print(f"  ERROR: Failed to process query: {e}")
            continue
    
    print(f"\nSuccessfully collected {len(plans)} query plans.")
    return collected_queries, plans, execution_times


def train_model(train_dataset: Neo4jQueryPlanDataset,
               val_dataset: Optional[Neo4jQueryPlanDataset],
               feature_extractor: FeatureExtractor,
               num_epochs: int = 100,
               batch_size: int = 32,
               learning_rate: float = 0.001,
               hidden_dim: int = 128,
               num_layers: int = 5,
               data_vec_dim: int = 32,
               checkpoint_path: str = 'neo4j_qpp_checkpoint.pt',
               device: str = 'cpu') -> PlanStructuredNetwork:
    """
    Train the plan-structured neural network.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        feature_extractor: Feature extractor instance
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim: Hidden dimension for neural units
        num_layers: Number of layers per neural unit
        data_vec_dim: Data vector dimension
        checkpoint_path: Path to save checkpoints
        device: 'cpu' or 'cuda'
        
    Returns:
        Trained model
    """
    # Create model
    model = PlanStructuredNetwork(
        feature_extractor=feature_extractor,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        data_vec_dim=data_vec_dim
    )
    
    # Initialize units from all training plans
    print("\nInitializing neural units from training data...")
    all_plans = [item['plan'] for item in train_dataset]
    model.initialize_units_from_plans(all_plans)
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Number of neural units: {len(model.units)}")
    
    # Create trainer
    trainer = PlanStructuredTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device
    )
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = trainer.train_epoch(train_dataset, batch_size=batch_size)
        print(f"Training Loss: {train_loss:.6f}")
        
        # Validate
        if val_dataset is not None:
            val_metrics = trainer.evaluate(val_dataset, batch_size=batch_size)
            val_loss = val_metrics['mse']
            trainer.val_losses.append(val_loss)
            
            print(f"Validation Loss: {val_loss:.6f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.6f}")
            print(f"Validation MAE: {val_metrics['mae']:.6f}")
            print(f"Validation R-value (mean): {val_metrics['mean_q_error']:.3f}")
            print(f"Validation R-value (median): {val_metrics['median_q_error']:.3f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(
                    checkpoint_path.replace('.pt', '_best.pt'),
                    epoch,
                    val_metrics
                )
                print("  Best model saved!")
        
        if (epoch + 1) % 10 == 0:
            metrics = val_metrics if val_dataset else {'train_loss': train_loss}
            trainer.save_checkpoint(
                checkpoint_path.replace('.pt', f'_epoch{epoch+1}.pt'),
                epoch,
                metrics
            )
    
    # Save final model
    final_metrics = trainer.evaluate(val_dataset if val_dataset else train_dataset)
    trainer.save_checkpoint(checkpoint_path, num_epochs, final_metrics)
    
    print("\n" + "=" * 50)
    print("Training completed!")
    print("=" * 50)
    
    return model
