"""Training script for Business Manager Regression Model

This script trains the hybrid regression model for manufacturing and resource optimization.
"""

import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegressionModel(nn.Module):
    """Hybrid regression model for business optimization"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize regression model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }, path)
        logger.info(f"Model checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model checkpoint loaded from {path}")


def generate_synthetic_business_data(
    n_samples: int = 10000,
    n_products: int = 10,
    n_features: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic business optimization data.
    
    Args:
        n_samples: Number of samples
        n_products: Number of products
        n_features: Number of features per product
        
    Returns:
        Tuple of (features, targets)
    """
    # Generate features
    # Features include: sales data, demand forecasts, costs, market conditions
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate targets (optimal production quantities)
    # Targets are influenced by features with some noise
    weights = np.random.randn(n_features, n_products)
    targets = np.dot(features, weights) + np.random.randn(n_samples, n_products) * 0.1
    
    # Ensure non-negative production quantities
    targets = np.maximum(targets, 0)
    
    return features, targets.astype(np.float32)


class RegressionTrainer:
    """Trainer for regression model"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        device: Optional[str] = None,
        use_mlflow: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            device: Device to train on
            use_mlflow: Whether to use MLflow logging
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mlflow = use_mlflow
        
        # Initialize model
        self.model = RegressionModel(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
        self.model.to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Scaler for features
        self.scaler = StandardScaler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': []
        }
        
        logger.info(f"Initialized RegressionTrainer on device: {self.device}")
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_split: float = 0.2,
        test_split: float = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders.
        
        Args:
            X: Features
            y: Targets
            val_split: Validation split ratio
            test_split: Test split ratio
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split/(1-test_split), random_state=42
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Prepared data: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_preds.append(predictions.detach().cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
        
        # Calculate RMSE
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        n_batches = len(train_loader)
        return {
            'loss': total_loss / n_batches,
            'rmse': rmse
        }
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        n_batches = len(val_loader)
        return total_loss / n_batches, rmse, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        checkpoint_dir: str = 'models/checkpoints/regression',
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
            
            run_name = f"regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            
            mlflow.log_params({
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'hidden_dims': self.hidden_dims,
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
            val_loss, val_rmse, val_metrics = self.validate(val_loader)
            
            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_rmse'].append(val_rmse)
            
            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_loss,
                    'train_rmse': train_metrics['rmse'],
                    'val_rmse': val_rmse,
                    'val_mae': val_metrics['mae'],
                    'val_r2': val_metrics['r2']
                }, step=epoch)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Val RMSE: {val_rmse:.6f}, "
                    f"Val R²: {val_metrics['r2']:.4f}"
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
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # RMSE plot
        axes[1].plot(self.history['train_rmse'], label='Train RMSE')
        axes[1].plot(self.history['val_rmse'], label='Val RMSE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Training and Validation RMSE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Training history plot saved to {save_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Regression Model')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of synthetic samples')
    parser.add_argument('--n-products', type=int, default=10, help='Number of products')
    
    # Model arguments
    parser.add_argument('--input-dim', type=int, default=20, help='Input dimension')
    parser.add_argument('--output-dim', type=int, default=10, help='Output dimension')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64], help='Hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--test-split', type=float, default=0.1, help='Test split')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints/regression',
                        help='Checkpoint directory')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow logging')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Regression Model Training")
    logger.info("="*60)
    
    # Load or generate data
    if args.synthetic or args.data_path is None:
        logger.info("Generating synthetic business data...")
        X, y = generate_synthetic_business_data(
            n_samples=args.n_samples,
            n_products=args.n_products,
            n_features=args.input_dim
        )
    else:
        logger.info(f"Loading data from {args.data_path}...")
        data = pd.read_csv(args.data_path)
        # Assume last n_products columns are targets
        X = data.iloc[:, :-args.output_dim].values.astype(np.float32)
        y = data.iloc[:, -args.output_dim:].values.astype(np.float32)
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Initialize trainer
    trainer = RegressionTrainer(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_mlflow=not args.no_mlflow
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        X, y,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping
    )
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_rmse, test_metrics = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Test RMSE: {test_rmse:.6f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.6f}")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    
    # Plot training history
    plot_path = Path(args.checkpoint_dir) / 'training_history.png'
    trainer.plot_training_history(str(plot_path))
    
    logger.info("="*60)
    logger.info("Training completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
