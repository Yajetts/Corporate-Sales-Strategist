"""
Training script for Sales Forecaster LSTM model

This script trains the LSTM model for sales trend prediction with
time-series cross-validation.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.sales_forecaster import SalesForecaster


def generate_synthetic_sales_data(
    n_timesteps: int = 1000,
    n_features: int = 5,
    trend: float = 0.1,
    seasonality_period: int = 30,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Generate synthetic sales data for training.
    
    Args:
        n_timesteps: Number of time steps
        n_features: Number of features
        trend: Trend component strength
        seasonality_period: Period of seasonal pattern
        noise_level: Noise level
        
    Returns:
        Synthetic time-series data
    """
    t = np.arange(n_timesteps)
    
    # Base sales with trend
    base = 100 + trend * t
    
    # Seasonal component
    seasonal = 20 * np.sin(2 * np.pi * t / seasonality_period)
    
    # Noise
    noise = noise_level * np.random.randn(n_timesteps) * base
    
    # Sales volume (main feature)
    sales = base + seasonal + noise
    sales = np.maximum(sales, 0)  # Ensure non-negative
    
    # Additional features
    data = np.zeros((n_timesteps, n_features))
    data[:, 0] = sales
    
    # Feature 2: Price (inversely correlated with sales)
    data[:, 1] = 100 - 0.2 * (sales - 100) + 5 * np.random.randn(n_timesteps)
    
    # Feature 3: Marketing spend (correlated with sales)
    data[:, 2] = 50 + 0.3 * (sales - 100) + 10 * np.random.randn(n_timesteps)
    
    # Feature 4: Competitor activity (random)
    data[:, 3] = 50 + 20 * np.sin(2 * np.pi * t / (seasonality_period * 1.5)) + 5 * np.random.randn(n_timesteps)
    
    # Feature 5: Market sentiment (trending)
    data[:, 4] = 50 + 0.05 * t + 10 * np.sin(2 * np.pi * t / (seasonality_period * 2)) + 5 * np.random.randn(n_timesteps)
    
    return data


def train_with_cross_validation(
    data: np.ndarray,
    n_splits: int = 5,
    **train_kwargs
) -> dict:
    """
    Train model with time-series cross-validation.
    
    Args:
        data: Time-series data
        n_splits: Number of CV splits
        **train_kwargs: Additional training arguments
        
    Returns:
        Cross-validation results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = {
        'fold_metrics': [],
        'models': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(data)):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")
        
        train_data = data[train_idx]
        val_data = data[val_idx]
        
        # Initialize model
        forecaster = SalesForecaster(
            input_size=data.shape[1],
            hidden_size=train_kwargs.get('hidden_size', 128),
            num_layers=train_kwargs.get('num_layers', 2),
            dropout=train_kwargs.get('dropout', 0.2),
            forecast_horizon=train_kwargs.get('forecast_horizon', 7),
            sequence_length=train_kwargs.get('sequence_length', 30)
        )
        
        # Train
        train_metrics = forecaster.train(
            train_data=train_data,
            val_data=val_data,
            epochs=train_kwargs.get('epochs', 100),
            batch_size=train_kwargs.get('batch_size', 32),
            learning_rate=train_kwargs.get('learning_rate', 0.001),
            checkpoint_dir=f"{train_kwargs.get('checkpoint_dir', 'models/checkpoints/sales_forecaster')}/fold_{fold+1}",
            early_stopping_patience=train_kwargs.get('early_stopping_patience', 10),
            verbose=train_kwargs.get('verbose', True)
        )
        
        # Evaluate on validation set
        val_metrics = forecaster.evaluate(val_data)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Train Loss: {train_metrics['final_train_loss']:.6f}")
        print(f"  Val Loss: {train_metrics['best_val_loss']:.6f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.6f}")
        print(f"  Val MAE: {val_metrics['mae']:.6f}")
        print(f"  Val R²: {val_metrics['r2_score']:.4f}")
        
        cv_results['fold_metrics'].append({
            'fold': fold + 1,
            'train_loss': train_metrics['final_train_loss'],
            'val_loss': train_metrics['best_val_loss'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2_score']
        })
        
        cv_results['models'].append(forecaster)
    
    # Calculate average metrics
    avg_metrics = {
        'avg_val_loss': np.mean([m['val_loss'] for m in cv_results['fold_metrics']]),
        'avg_val_rmse': np.mean([m['val_rmse'] for m in cv_results['fold_metrics']]),
        'avg_val_mae': np.mean([m['val_mae'] for m in cv_results['fold_metrics']]),
        'avg_val_r2': np.mean([m['val_r2'] for m in cv_results['fold_metrics']]),
        'std_val_rmse': np.std([m['val_rmse'] for m in cv_results['fold_metrics']])
    }
    
    cv_results['average_metrics'] = avg_metrics
    
    print(f"\n{'='*50}")
    print("Cross-Validation Summary")
    print(f"{'='*50}")
    print(f"Average Val RMSE: {avg_metrics['avg_val_rmse']:.6f} ± {avg_metrics['std_val_rmse']:.6f}")
    print(f"Average Val MAE: {avg_metrics['avg_val_mae']:.6f}")
    print(f"Average Val R²: {avg_metrics['avg_val_r2']:.4f}")
    
    return cv_results


def plot_training_history(forecaster: SalesForecaster, save_path: str):
    """Plot training history."""
    history = forecaster.training_history
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_loss'], label='Train Loss', marker='o')
    if history['val_loss']:
        plt.plot(history['epochs'], history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_predictions(
    forecaster: SalesForecaster,
    test_data: np.ndarray,
    save_path: str,
    n_samples: int = 5
):
    """Plot sample predictions."""
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    
    if n_samples == 1:
        axes = [axes]
    
    sequence_length = forecaster.sequence_length
    forecast_horizon = forecaster.forecast_horizon
    
    for i, ax in enumerate(axes):
        # Get a sample
        start_idx = i * 50
        if start_idx + sequence_length + forecast_horizon > len(test_data):
            break
        
        input_seq = test_data[start_idx:start_idx + sequence_length]
        actual_future = test_data[start_idx + sequence_length:start_idx + sequence_length + forecast_horizon, 0]
        
        # Predict
        prediction = forecaster.predict(input_seq, return_confidence=True)
        
        # Plot
        x_hist = np.arange(sequence_length)
        x_future = np.arange(sequence_length, sequence_length + forecast_horizon)
        
        ax.plot(x_hist, input_seq[:, 0], label='Historical', color='blue', linewidth=2)
        ax.plot(x_future, actual_future, label='Actual', color='green', linewidth=2, marker='o')
        ax.plot(x_future, prediction['forecast'], label='Predicted', color='red', linewidth=2, marker='s')
        
        if 'confidence_interval' in prediction:
            ax.fill_between(
                x_future,
                prediction['confidence_interval']['lower'],
                prediction['confidence_interval']['upper'],
                alpha=0.3,
                color='red',
                label='90% CI'
            )
        
        ax.axvline(x=sequence_length, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Sales Volume')
        ax.set_title(f'Sample {i+1}: Sales Forecast')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Predictions plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Sales Forecaster LSTM')
    parser.add_argument('--data-path', type=str, help='Path to training data CSV')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--n-timesteps', type=int, default=1000, help='Number of timesteps for synthetic data')
    parser.add_argument('--sequence-length', type=int, default=30, help='Input sequence length')
    parser.add_argument('--forecast-horizon', type=int, default=7, help='Forecast horizon')
    parser.add_argument('--hidden-size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--cv-splits', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints/sales_forecaster',
                        help='Checkpoint directory')
    parser.add_argument('--no-cv', action='store_true', help='Skip cross-validation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Sales Forecaster LSTM Training")
    print("="*60)
    
    # Load or generate data
    if args.synthetic or args.data_path is None:
        print("\nGenerating synthetic sales data...")
        data = generate_synthetic_sales_data(n_timesteps=args.n_timesteps)
        print(f"Generated data shape: {data.shape}")
    else:
        print(f"\nLoading data from {args.data_path}...")
        df = pd.read_csv(args.data_path)
        data = df.values
        print(f"Loaded data shape: {data.shape}")
    
    # Split data
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_data)} timesteps")
    print(f"  Val: {len(val_data)} timesteps")
    print(f"  Test: {len(test_data)} timesteps")
    
    # Training parameters
    train_kwargs = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'forecast_horizon': args.forecast_horizon,
        'sequence_length': args.sequence_length,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'checkpoint_dir': args.checkpoint_dir,
        'early_stopping_patience': 10,
        'verbose': True
    }
    
    if args.no_cv:
        # Train single model
        print("\nTraining single model...")
        forecaster = SalesForecaster(
            input_size=data.shape[1],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            forecast_horizon=args.forecast_horizon,
            sequence_length=args.sequence_length
        )
        
        train_metrics = forecaster.train(
            train_data=train_data,
            val_data=val_data,
            **{k: v for k, v in train_kwargs.items() if k not in ['hidden_size', 'num_layers', 'dropout', 'forecast_horizon', 'sequence_length']}
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = forecaster.evaluate(test_data)
        print(f"Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"Test MAE: {test_metrics['mae']:.6f}")
        print(f"Test R²: {test_metrics['r2_score']:.4f}")
        
        # Plot results
        plot_training_history(forecaster, os.path.join(args.checkpoint_dir, 'training_history.png'))
        plot_predictions(forecaster, test_data, os.path.join(args.checkpoint_dir, 'predictions.png'))
        
    else:
        # Cross-validation
        print("\nStarting time-series cross-validation...")
        cv_results = train_with_cross_validation(
            data=train_data,
            n_splits=args.cv_splits,
            **train_kwargs
        )
        
        # Select best model and evaluate on test set
        best_fold = np.argmin([m['val_loss'] for m in cv_results['fold_metrics']])
        best_forecaster = cv_results['models'][best_fold]
        
        print(f"\nBest model: Fold {best_fold + 1}")
        print("Evaluating on test set...")
        test_metrics = best_forecaster.evaluate(test_data)
        print(f"Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"Test MAE: {test_metrics['mae']:.6f}")
        print(f"Test R²: {test_metrics['r2_score']:.4f}")
        
        # Save best model
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        best_forecaster.save_checkpoint(best_model_path)
        print(f"\nBest model saved to {best_model_path}")
        
        # Plot results
        plot_training_history(best_forecaster, os.path.join(args.checkpoint_dir, 'training_history.png'))
        plot_predictions(best_forecaster, test_data, os.path.join(args.checkpoint_dir, 'predictions.png'))
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == '__main__':
    main()
