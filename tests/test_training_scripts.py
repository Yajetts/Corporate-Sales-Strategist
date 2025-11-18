"""Smoke tests for training scripts

These tests verify that training scripts can be executed without errors.
They use minimal data and epochs to ensure quick execution.
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path
import tempfile
import shutil


class TestTrainingScripts:
    """Smoke tests for training script execution"""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_train_autoencoder_synthetic_data(self, temp_checkpoint_dir):
        """Test autoencoder training script with synthetic data"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_autoencoder',
            '--synthetic',
            '--n-samples', '100',
            '--input-dim', '64',
            '--latent-dim', '16',
            '--hidden-dims', '32',
            '--num-epochs', '2',
            '--batch-size', '32',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        # Set PYTHONPATH to include project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        # Check script executed successfully
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check checkpoint files were created
        checkpoint_path = Path(temp_checkpoint_dir)
        assert (checkpoint_path / 'final_model.pt').exists()
    
    def test_train_gnn_synthetic_data(self, temp_checkpoint_dir):
        """Test GNN training script with synthetic data"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_gnn',
            '--synthetic',
            '--n-graphs', '20',
            '--in-channels', '32',
            '--hidden-channels', '64',
            '--out-channels', '32',
            '--num-layers', '2',
            '--num-epochs', '2',
            '--batch-size', '8',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        # Set PYTHONPATH to include project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        # Check script executed successfully
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check checkpoint files were created
        checkpoint_path = Path(temp_checkpoint_dir)
        assert (checkpoint_path / 'final_model.pt').exists()
    
    def test_train_regression_synthetic_data(self, temp_checkpoint_dir):
        """Test regression training script with synthetic data"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_regression',
            '--synthetic',
            '--n-samples', '200',
            '--n-products', '5',
            '--input-dim', '10',
            '--output-dim', '5',
            '--hidden-dims', '32', '16',
            '--num-epochs', '2',
            '--batch-size', '32',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        # Set PYTHONPATH to include project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        # Check script executed successfully
        assert result.returncode == 0, f"Script failed with error: {result.stderr}"
        
        # Check checkpoint files were created
        checkpoint_path = Path(temp_checkpoint_dir)
        assert (checkpoint_path / 'final_model.pt').exists()
    
    def test_train_autoencoder_help(self):
        """Test autoencoder training script help message"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_autoencoder',
            '--help'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )
        
        assert result.returncode == 0
        assert 'Train Autoencoder' in result.stdout
    
    def test_train_gnn_help(self):
        """Test GNN training script help message"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_gnn',
            '--help'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )
        
        assert result.returncode == 0
        assert 'Train GNN' in result.stdout
    
    def test_train_regression_help(self):
        """Test regression training script help message"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_regression',
            '--help'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            env=env
        )
        
        assert result.returncode == 0
        assert 'Train Regression Model' in result.stdout
    
    def test_train_autoencoder_with_custom_params(self, temp_checkpoint_dir):
        """Test autoencoder with custom hyperparameters"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_autoencoder',
            '--synthetic',
            '--n-samples', '50',
            '--input-dim', '128',
            '--latent-dim', '32',
            '--hidden-dims', '64', '48',
            '--learning-rate', '0.001',
            '--num-epochs', '1',
            '--batch-size', '16',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        assert result.returncode == 0
    
    def test_train_gnn_with_custom_params(self, temp_checkpoint_dir):
        """Test GNN with custom hyperparameters"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_gnn',
            '--synthetic',
            '--n-graphs', '10',
            '--in-channels', '16',
            '--hidden-channels', '32',
            '--out-channels', '16',
            '--num-layers', '3',
            '--dropout', '0.3',
            '--learning-rate', '0.001',
            '--num-epochs', '1',
            '--batch-size', '4',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        assert result.returncode == 0
    
    def test_train_regression_with_custom_params(self, temp_checkpoint_dir):
        """Test regression with custom hyperparameters"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_regression',
            '--synthetic',
            '--n-samples', '100',
            '--n-products', '3',
            '--input-dim', '15',
            '--output-dim', '3',
            '--hidden-dims', '64', '32', '16',
            '--dropout', '0.1',
            '--learning-rate', '0.001',
            '--num-epochs', '1',
            '--batch-size', '16',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        assert result.returncode == 0


class TestTrainingScriptOutputs:
    """Test training script outputs and artifacts"""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary directory for checkpoints"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    def test_autoencoder_creates_training_plot(self, temp_checkpoint_dir):
        """Test that autoencoder creates training history plot"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_autoencoder',
            '--synthetic',
            '--n-samples', '50',
            '--num-epochs', '2',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        subprocess.run(cmd, capture_output=True, timeout=60, env=env)
        
        # Check plot was created
        plot_path = Path(temp_checkpoint_dir) / 'training_history.png'
        assert plot_path.exists()
    
    def test_gnn_creates_training_plot(self, temp_checkpoint_dir):
        """Test that GNN creates training history plot"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_gnn',
            '--synthetic',
            '--n-graphs', '10',
            '--num-epochs', '2',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        subprocess.run(cmd, capture_output=True, timeout=60, env=env)
        
        # Check plot was created
        plot_path = Path(temp_checkpoint_dir) / 'training_history.png'
        assert plot_path.exists()
    
    def test_regression_creates_training_plot(self, temp_checkpoint_dir):
        """Test that regression creates training history plot"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_regression',
            '--synthetic',
            '--n-samples', '100',
            '--num-epochs', '2',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        subprocess.run(cmd, capture_output=True, timeout=60, env=env)
        
        # Check plot was created
        plot_path = Path(temp_checkpoint_dir) / 'training_history.png'
        assert plot_path.exists()
    
    def test_autoencoder_creates_best_model(self, temp_checkpoint_dir):
        """Test that autoencoder saves best model checkpoint"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_autoencoder',
            '--synthetic',
            '--n-samples', '50',
            '--num-epochs', '3',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        subprocess.run(cmd, capture_output=True, timeout=60, env=env)
        
        # Check best model was saved
        best_model_path = Path(temp_checkpoint_dir) / 'best_model.pt'
        assert best_model_path.exists()
    
    def test_gnn_creates_best_model(self, temp_checkpoint_dir):
        """Test that GNN saves best model checkpoint"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_gnn',
            '--synthetic',
            '--n-graphs', '10',
            '--num-epochs', '3',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        subprocess.run(cmd, capture_output=True, timeout=60, env=env)
        
        # Check best model was saved
        best_model_path = Path(temp_checkpoint_dir) / 'best_model.pt'
        assert best_model_path.exists()
    
    def test_regression_creates_best_model(self, temp_checkpoint_dir):
        """Test that regression saves best model checkpoint"""
        cmd = [
            sys.executable,
            '-m', 'src.training.train_regression',
            '--synthetic',
            '--n-samples', '100',
            '--num-epochs', '3',
            '--checkpoint-dir', temp_checkpoint_dir,
            '--no-mlflow'
        ]
        
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        subprocess.run(cmd, capture_output=True, timeout=60, env=env)
        
        # Check best model was saved
        best_model_path = Path(temp_checkpoint_dir) / 'best_model.pt'
        assert best_model_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
