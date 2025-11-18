"""Training script for Graph Neural Network

This script trains the GNN for relationship modeling in the Market Decipherer module.
"""

import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

from src.models.gnn import GraphNeuralNetwork
from src.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_graph_data(
    n_graphs: int = 100,
    n_nodes_range: Tuple[int, int] = (50, 200),
    n_features: int = 64,
    edge_prob: float = 0.1
) -> list:
    """
    Generate synthetic graph data for training.
    
    Args:
        n_graphs: Number of graphs to generate
        n_nodes_range: Range of nodes per graph
        n_features: Number of node features
        edge_prob: Probability of edge creation
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    graphs = []
    
    for _ in range(n_graphs):
        n_nodes = np.random.randint(n_nodes_range[0], n_nodes_range[1])
        
        # Generate random graph
        G = nx.erdos_renyi_graph(n_nodes, edge_prob)
        
        # Add node features
        for node in G.nodes():
            G.nodes[node]['x'] = np.random.randn(n_features).astype(np.float32)
        
        # Convert to PyTorch Geometric format
        data = from_networkx(G)
        
        # Stack node features
        if hasattr(data, 'x') and data.x is not None:
            data.x = torch.stack([data.x[i] for i in range(len(data.x))])
        else:
            data.x = torch.randn(n_nodes, n_features)
        
        # Add dummy labels for link prediction
        data.y = torch.randint(0, 2, (data.edge_index.size(1),))
        
        graphs.append(data)
    
    return graphs


class GNNTrainer:
    """Trainer for Graph Neural Network"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: Optional[str] = None,
        use_mlflow: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            out_channels: Output embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            device: Device to train on
            use_mlflow: Whether to use MLflow logging
        """
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mlflow = use_mlflow
        
        # Initialize model
        self.model = GraphNeuralNetwork(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout
        )
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logger.info(f"Initialized GNNTrainer on device: {self.device}")
    
    def prepare_data(
        self,
        graphs: list,
        val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders.
        
        Args:
            graphs: List of graph data
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Split data
        n_val = int(len(graphs) * val_split)
        n_train = len(graphs) - n_val
        
        train_graphs = graphs[:n_train]
        val_graphs = graphs[n_train:]
        
        # Create data loaders
        train_loader = DataLoader(
            train_graphs,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_graphs,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        logger.info(f"Prepared data: train={len(train_graphs)}, val={len(val_graphs)}")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            embeddings = self.model(batch.x, batch.edge_index, batch.batch)
            
            # For link prediction task
            # Sample positive and negative edges
            edge_index = batch.edge_index
            num_edges = edge_index.size(1)
            
            # Positive edges
            pos_src = embeddings[edge_index[0]]
            pos_dst = embeddings[edge_index[1]]
            pos_scores = (pos_src * pos_dst).sum(dim=1)
            
            # Negative edges (random sampling)
            neg_dst_idx = torch.randint(0, embeddings.size(0), (num_edges,), device=self.device)
            neg_dst = embeddings[neg_dst_idx]
            neg_scores = (pos_src * neg_dst).sum(dim=1)
            
            # Compute loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores,
                torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores,
                torch.zeros_like(neg_scores)
            )
            loss = pos_loss + neg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Calculate accuracy
            pos_pred = (torch.sigmoid(pos_scores) > 0.5).float()
            neg_pred = (torch.sigmoid(neg_scores) > 0.5).float()
            correct = (pos_pred == 1).sum() + (neg_pred == 0).sum()
            total_correct += correct.item()
            total_samples += num_edges * 2
        
        n_batches = len(train_loader)
        return {
            'loss': total_loss / n_batches,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0
        }
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                embeddings = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Link prediction
                edge_index = batch.edge_index
                num_edges = edge_index.size(1)
                
                pos_src = embeddings[edge_index[0]]
                pos_dst = embeddings[edge_index[1]]
                pos_scores = (pos_src * pos_dst).sum(dim=1)
                
                neg_dst_idx = torch.randint(0, embeddings.size(0), (num_edges,), device=self.device)
                neg_dst = embeddings[neg_dst_idx]
                neg_scores = (pos_src * neg_dst).sum(dim=1)
                
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_scores,
                    torch.ones_like(pos_scores)
                )
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_scores,
                    torch.zeros_like(neg_scores)
                )
                loss = pos_loss + neg_loss
                
                total_loss += loss.item()
                
                # Calculate accuracy
                pos_pred = (torch.sigmoid(pos_scores) > 0.5).float()
                neg_pred = (torch.sigmoid(neg_scores) > 0.5).float()
                correct = (pos_pred == 1).sum() + (neg_pred == 0).sum()
                total_correct += correct.item()
                total_samples += num_edges * 2
        
        n_batches = len(val_loader)
        return total_loss / n_batches, total_correct / total_samples if total_samples > 0 else 0.0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        checkpoint_dir: str = 'models/checkpoints/gnn',
        early_stopping_patience: int = 10
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            checkpoint_dir: Directory for checkpoints
            early_stopping_patience: Early stopping patience
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        if self.use_mlflow:
            mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
            
            run_name = f"gnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            
            mlflow.log_params({
                'in_channels': self.in_channels,
                'hidden_channels': self.hidden_channels,
                'out_channels': self.out_channels,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': num_epochs
            })
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_acc)
            
            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_loss,
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_acc
                }, step=epoch)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Acc: {val_acc:.4f}"
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                best_model_path = checkpoint_path / 'best_model.pt'
                self.model.save_checkpoint(str(best_model_path))
                logger.info(f"Saved best model with val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save final model
        final_model_path = checkpoint_path / 'final_model.pt'
        self.model.save_checkpoint(str(final_model_path))
        
        # Log model to MLflow
        if self.use_mlflow:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()
        
        logger.info("Training completed!")
        return self.history
    
    def plot_training_history(self, save_path: str):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Accuracy')
        axes[1].plot(self.history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Training history plot saved to {save_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train GNN')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--n-graphs', type=int, default=100, help='Number of synthetic graphs')
    
    # Model arguments
    parser.add_argument('--in-channels', type=int, default=64, help='Input feature dimension')
    parser.add_argument('--hidden-channels', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--out-channels', type=int, default=64, help='Output embedding dimension')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints/gnn',
                        help='Checkpoint directory')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow logging')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("GNN Training")
    logger.info("="*60)
    
    # Load or generate data
    if args.synthetic or args.data_path is None:
        logger.info("Generating synthetic graph data...")
        graphs = generate_synthetic_graph_data(
            n_graphs=args.n_graphs,
            n_features=args.in_channels
        )
    else:
        logger.info(f"Loading data from {args.data_path}...")
        # Implement custom data loading logic here
        raise NotImplementedError("Custom data loading not implemented")
    
    logger.info(f"Number of graphs: {len(graphs)}")
    
    # Initialize trainer
    trainer = GNNTrainer(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_mlflow=not args.no_mlflow
    )
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(graphs, val_split=args.val_split)
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping
    )
    
    # Plot training history
    plot_path = Path(args.checkpoint_dir) / 'training_history.png'
    trainer.plot_training_history(str(plot_path))
    
    logger.info("="*60)
    logger.info("Training completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
