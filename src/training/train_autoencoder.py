"""Training script for Autoencoder model

This script trains the Variational Autoencoder for dimensionality reduction
in the Market Decipherer module.
"""

import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.autoencoder import VariationalAutoencoder
from src.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_market_data(
    n_samples: int = 10000,
    n_features: int = 512
) -> np.ndarray:
    """
    Generate synthetic market data for training.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        Synthetic market data
    """
    # Generate data with some structure
    n_clusters = 5
    samples_per_cluster = n_samples // n_clusters
    
    data = []
    for i in range(n_clusters):
        # Create cluster center
        center = np.random.randn(n_features) * 10
        # Generate samples around center
        cluster_data = center + np.random.randn(samples_per_cluster, n_features) * 2
        data.append(cluster_data)
    
    data = np.vstack(data)
    np.random.shuffle(data)
    
    return data.astype(np.float32)


class AutoencoderTrainer:
    """Trainer for Variational Autoencoder"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: list = None,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        device: Optional[str] = None,
        use_mlflow: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            input_dim: Input dimension
            latent_dim: Latent dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            batch_size: Batch size
            device: Device to train on
            use_mlflow: Whether to use MLflow logging
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mlflow = use_mlflow
        
        # Initialize model
        self.model = VariationalAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=self.hidden_dims
        )
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'recon_loss': [],
            'kl_loss': []
        }
        
        logger.info(f"Initialized AutoencoderTrainer on device: {self.device}")
    
    def prepare_data(
        self,
        data: np.ndarray,
        val_split: float = 0.2
    ) -> tuple:
        """
        Prepare data loaders.
        
        Args:
            data: Training data
            val_split: Validation split ratio
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Split data
        n_val = int(len(data) * val_split)
        n_train = len(data) - n_val
        
        train_data = data[:n_train]
        val_data = data[n_train:]
        
        # Create datasets
        train_dataset = TensorDataset(torch.FloatTensor(train_data))
        val_dataset = TensorDataset(torch.FloatTensor(val_data))
        
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
        
        logger.info(f"Prepared data: train={len(train_data)}, val={len(val_data)}")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        for batch in train_loader:
            x = batch[0].to(self.device)
            
            # Forward pass
            recon_x, mu, logvar = self.model(x)
            loss, recon_loss, kl_loss = self.model.loss_function(recon_x, x, mu, logvar)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        n_batches = len(train_loader)
        return {
            'loss': total_loss / n_batches,
            'recon_loss': total_recon / n_batches,
            'kl_loss': total_kl / n_batches
        }
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                recon_x, mu, logvar = self.model(x)
                loss, _, _ = self.model.loss_function(recon_x, x, mu, logvar)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        checkpoint_dir: str = 'models/checkpoints/autoencoder',
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
            
            run_name = f"autoencoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            
            mlflow.log_params({
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'hidden_dims': self.hidden_dims,
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
            val_loss = self.validate(val_loader)
            
            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_loss)
            self.history['recon_loss'].append(train_metrics['recon_loss'])
            self.history['kl_loss'].append(train_metrics['kl_loss'])
            
            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_loss,
                    'recon_loss': train_metrics['recon_loss'],
                    'kl_loss': train_metrics['kl_loss']
                }, step=epoch)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"Recon: {train_metrics['recon_loss']:.6f}, "
                    f"KL: {train_metrics['kl_loss']:.6f}"
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
        
        # Component losses
        axes[1].plot(self.history['recon_loss'], label='Reconstruction Loss')
        axes[1].plot(self.history['kl_loss'], label='KL Divergence')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Training history plot saved to {save_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Autoencoder')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, help='Path to training data')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of synthetic samples')
    
    # Model arguments
    parser.add_argument('--input-dim', type=int, default=512, help='Input dimension')
    parser.add_argument('--latent-dim', type=int, default=64, help='Latent dimension')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128], help='Hidden dimensions')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--early-stopping', type=int, default=10, help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints/autoencoder',
                        help='Checkpoint directory')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow logging')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Autoencoder Training")
    logger.info("="*60)
    
    # Load or generate data
    if args.synthetic or args.data_path is None:
        logger.info("Generating synthetic market data...")
        data = generate_synthetic_market_data(
            n_samples=args.n_samples,
            n_features=args.input_dim
        )
    else:
        logger.info(f"Loading data from {args.data_path}...")
        data = np.load(args.data_path)
    
    logger.info(f"Data shape: {data.shape}")
    
    # Initialize trainer
    trainer = AutoencoderTrainer(
        input_dim=data.shape[1],
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_mlflow=not args.no_mlflow
    )
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(data, val_split=args.val_split)
    
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
