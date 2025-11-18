"""Training visualization utilities

This module provides utilities for visualizing:
- Training progress and metrics
- Loss curves and convergence
- Model performance comparisons
- Hyperparameter tuning results
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class TrainingVisualizer:
    """Visualizer for training metrics and progress"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir) if output_dir else Path('visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized TrainingVisualizer with output dir: {self.output_dir}")
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot training history with multiple metrics.
        
        Args:
            history: Dictionary with metric names as keys and lists of values
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            Matplotlib Figure
        """
        n_metrics = len(history)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        for idx, (metric_name, values) in enumerate(history.items()):
            ax = axes[idx] if n_metrics > 1 else axes[0]
            
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker='o', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_loss_curves(
        self,
        train_loss: List[float],
        val_loss: Optional[List[float]] = None,
        title: str = "Loss Curves",
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot training and validation loss curves.
        
        Args:
            train_loss: Training loss values
            val_loss: Validation loss values
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2)
        
        if val_loss:
            ax.plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_metric_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        title: str = "Model Comparison",
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot comparison of metrics across different models/runs.
        
        Args:
            metrics_dict: Dictionary mapping model names to metric dictionaries
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            Matplotlib Figure
        """
        # Convert to DataFrame
        df = pd.DataFrame(metrics_dict).T
        
        n_metrics = len(df.columns)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        for idx, metric in enumerate(df.columns):
            ax = axes[idx] if n_metrics > 1 else axes[0]
            
            df[metric].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Model', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_learning_rate_schedule(
        self,
        learning_rates: List[float],
        title: str = "Learning Rate Schedule",
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot learning rate schedule over training.
        
        Args:
            learning_rates: Learning rate values
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        steps = range(len(learning_rates))
        ax.plot(steps, learning_rates, linewidth=2, color='darkorange')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: Names of classes
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot predictions vs actual values scatter plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig


class HyperparameterVisualizer:
    """Visualizer for hyperparameter tuning results"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir) if output_dir else Path('visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized HyperparameterVisualizer with output dir: {self.output_dir}")
    
    def plot_hyperparameter_importance(
        self,
        param_importance: Dict[str, float],
        title: str = "Hyperparameter Importance",
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot hyperparameter importance.
        
        Args:
            param_importance: Dictionary mapping parameter names to importance scores
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        params, importance = zip(*sorted_params)
        
        ax.barh(params, importance, color='steelblue')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Hyperparameter', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_hyperparameter_heatmap(
        self,
        results_df: pd.DataFrame,
        param1: str,
        param2: str,
        metric: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ) -> Figure:
        """
        Plot heatmap of metric values for two hyperparameters.
        
        Args:
            results_df: DataFrame with hyperparameter tuning results
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to visualize
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            Matplotlib Figure
        """
        # Pivot data for heatmap
        pivot_data = results_df.pivot(index=param2, columns=param1, values=metric)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.4f',
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': metric}
        )
        
        if title is None:
            title = f'{metric} vs {param1} and {param2}'
        
        ax.set_xlabel(param1, fontsize=12)
        ax.set_ylabel(param2, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_parallel_coordinates(
        self,
        results_df: pd.DataFrame,
        params: List[str],
        metric: str,
        title: str = "Hyperparameter Parallel Coordinates",
        save_name: Optional[str] = None
    ):
        """
        Create interactive parallel coordinates plot for hyperparameter tuning.
        
        Args:
            results_df: DataFrame with hyperparameter tuning results
            params: List of parameter names to include
            metric: Metric column name
            title: Plot title
            save_name: Filename to save plot (HTML)
        """
        # Prepare data
        plot_df = results_df[params + [metric]].copy()
        
        # Create figure
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=plot_df[metric],
                    colorscale='Viridis',
                    showscale=True,
                    cmin=plot_df[metric].min(),
                    cmax=plot_df[metric].max()
                ),
                dimensions=[
                    dict(
                        label=param,
                        values=plot_df[param]
                    ) for param in params
                ] + [
                    dict(
                        label=metric,
                        values=plot_df[metric]
                    )
                ]
            )
        )
        
        fig.update_layout(
            title=title,
            font=dict(size=12)
        )
        
        if save_name:
            save_path = self.output_dir / save_name
            fig.write_html(str(save_path))
            logger.info(f"Saved interactive plot to {save_path}")
        
        return fig


def create_training_dashboard(
    history: Dict[str, List[float]],
    metrics: Dict[str, float],
    model_name: str,
    output_dir: str = 'visualizations'
):
    """
    Create a comprehensive training dashboard with multiple plots.
    
    Args:
        history: Training history dictionary
        metrics: Final metrics dictionary
        model_name: Name of the model
        output_dir: Output directory for plots
    """
    visualizer = TrainingVisualizer(output_dir=output_dir)
    
    # Create output directory for this model
    model_dir = visualizer.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training history
    visualizer.plot_training_history(
        history=history,
        title=f"{model_name} Training History",
        save_name=f"{model_name}/training_history.png"
    )
    
    # Plot loss curves if available
    if 'train_loss' in history:
        val_loss = history.get('val_loss')
        visualizer.plot_loss_curves(
            train_loss=history['train_loss'],
            val_loss=val_loss,
            title=f"{model_name} Loss Curves",
            save_name=f"{model_name}/loss_curves.png"
        )
    
    # Create metrics summary
    fig, ax = plt.subplots(figsize=(8, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    ax.barh(metric_names, metric_values, color='steelblue')
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title(f"{model_name} Final Metrics", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(model_dir / 'final_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created training dashboard for {model_name} in {model_dir}")


if __name__ == '__main__':
    # Example usage
    visualizer = TrainingVisualizer(output_dir='test_visualizations')
    
    # Example training history
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'val_loss': [0.55, 0.45, 0.35, 0.3, 0.28],
        'train_accuracy': [0.7, 0.75, 0.8, 0.85, 0.88],
        'val_accuracy': [0.68, 0.73, 0.78, 0.82, 0.85]
    }
    
    visualizer.plot_training_history(history, save_name='example_history.png')
    visualizer.plot_loss_curves(
        history['train_loss'],
        history['val_loss'],
        save_name='example_loss.png'
    )
    
    print("Example visualizations created!")
