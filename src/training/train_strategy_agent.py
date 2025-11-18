"""
Training script for the Strategy Agent (RL)

This script trains the PPO agent on the market simulation environment
and logs metrics to MLflow.
"""

import os
import argparse
from datetime import datetime
import mlflow
import mlflow.pytorch

from src.models.strategy_agent import StrategyAgent
from src.utils.config import Config


def train_strategy_agent(
    total_timesteps: int = 100000,
    num_competitors: int = 5,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    checkpoint_dir: str = None,
    experiment_name: str = "strategy_agent_training"
):
    """
    Train the strategy agent and log to MLflow.
    
    Args:
        total_timesteps: Total training timesteps
        num_competitors: Number of competitors in simulation
        learning_rate: Learning rate for PPO
        batch_size: Batch size for training
        checkpoint_dir: Directory for checkpoints
        experiment_name: MLflow experiment name
    """
    # Set up checkpoint directory
    if checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = f"models/checkpoints/strategy_agent_{timestamp}"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'total_timesteps': total_timesteps,
            'num_competitors': num_competitors,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'algorithm': 'PPO'
        })
        
        # Initialize agent
        print("Initializing Strategy Agent...")
        agent = StrategyAgent(num_competitors=num_competitors)
        
        # Train agent
        print(f"Training agent for {total_timesteps} timesteps...")
        training_results = agent.train(
            total_timesteps=total_timesteps,
            checkpoint_dir=checkpoint_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=1
        )
        
        # Log metrics
        eval_metrics = training_results['eval_metrics']
        mlflow.log_metrics({
            'mean_reward': eval_metrics['mean_reward'],
            'std_reward': eval_metrics['std_reward'],
            'mean_revenue': eval_metrics['mean_revenue'],
            'std_revenue': eval_metrics['std_revenue'],
            'mean_sales': eval_metrics['mean_sales'],
            'std_sales': eval_metrics['std_sales']
        })
        
        # Log model
        model_path = training_results['final_model_path']
        mlflow.log_artifact(model_path)
        
        # Register model in MLflow
        mlflow.log_param('model_path', model_path)
        
        print(f"\nTraining completed!")
        print(f"Model saved to: {model_path}")
        print(f"\nEvaluation Metrics:")
        print(f"  Mean Reward: {eval_metrics['mean_reward']:.4f} ± {eval_metrics['std_reward']:.4f}")
        print(f"  Mean Revenue: ${eval_metrics['mean_revenue']:.2f} ± ${eval_metrics['std_revenue']:.2f}")
        print(f"  Mean Sales: {eval_metrics['mean_sales']:.2f} ± {eval_metrics['std_sales']:.2f}")
        
        return model_path, eval_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Strategy Agent')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--competitors', type=int, default=5,
                        help='Number of competitors')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Checkpoint directory')
    parser.add_argument('--experiment', type=str, default='strategy_agent_training',
                        help='MLflow experiment name')
    
    args = parser.parse_args()
    
    train_strategy_agent(
        total_timesteps=args.timesteps,
        num_competitors=args.competitors,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment
    )


if __name__ == '__main__':
    main()
