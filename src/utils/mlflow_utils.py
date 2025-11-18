"""MLflow utilities for model tracking, registry, and management

This module provides utilities for:
- Model versioning and registry
- Experiment tracking
- Model loading from MLflow
- Model comparison and selection
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd

from src.utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowManager:
    """Manager for MLflow operations"""
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize MLflow manager.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Experiment name
        """
        self.tracking_uri = tracking_uri or Config.MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name or Config.MLFLOW_EXPERIMENT_NAME
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Initialize client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                self.experiment = mlflow.get_experiment(self.experiment_id)
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            logger.warning(f"Could not access MLflow experiment: {e}")
            self.experiment_id = None
            self.experiment = None
        
        logger.info(f"Initialized MLflowManager with tracking URI: {self.tracking_uri}")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Tags to add to the run
            
        Returns:
            Active MLflow run
        """
        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run"""
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics to current run"""
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact to current run"""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None
    ):
        """
        Log PyTorch model to MLflow.
        
        Args:
            model: PyTorch model
            artifact_path: Path within run artifacts
            registered_model_name: Name for model registry
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"Logged model to {artifact_path}")
    
    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> Any:
        """
        Register model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model (e.g., runs:/<run_id>/model)
            name: Name for registered model
            tags: Tags for the model version
            description: Description of the model
            
        Returns:
            ModelVersion object
        """
        try:
            # Register model
            model_version = mlflow.register_model(model_uri, name)
            
            # Add tags if provided
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=name,
                        version=model_version.version,
                        key=key,
                        value=value
                    )
            
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model '{name}' version {model_version.version}")
            return model_version
        
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Any:
        """
        Load model from MLflow Model Registry.
        
        Args:
            model_name: Name of registered model
            version: Specific version number (e.g., "1", "2")
            stage: Stage name (e.g., "Production", "Staging")
            
        Returns:
            Loaded model
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                # Load latest version
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def get_model_versions(
        self,
        model_name: str,
        stages: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Get all versions of a registered model.
        
        Args:
            model_name: Name of registered model
            stages: Filter by stages (e.g., ["Production", "Staging"])
            
        Returns:
            List of ModelVersion objects
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if stages:
                versions = [v for v in versions if v.current_stage in stages]
            
            logger.info(f"Found {len(versions)} versions for model '{model_name}'")
            return versions
        
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Transition model version to a new stage.
        
        Args:
            model_name: Name of registered model
            version: Version number
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing: Whether to archive existing versions in target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing
            )
            logger.info(f"Transitioned model '{model_name}' v{version} to {stage}")
        
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
    
    def search_runs(
        self,
        filter_string: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        max_results: int = 100
    ) -> List[Any]:
        """
        Search for runs in the experiment.
        
        Args:
            filter_string: Filter query (e.g., "metrics.rmse < 0.5")
            order_by: List of order by clauses (e.g., ["metrics.rmse ASC"])
            max_results: Maximum number of results
            
        Returns:
            List of Run objects
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results,
                run_view_type=ViewType.ACTIVE_ONLY
            )
            logger.info(f"Found {len(runs)} runs")
            return runs
        
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def get_best_run(
        self,
        metric_name: str,
        ascending: bool = True,
        filter_string: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Name of metric to optimize
            ascending: Whether lower is better
            filter_string: Optional filter query
            
        Returns:
            Best Run object or None
        """
        order = "ASC" if ascending else "DESC"
        order_by = [f"metrics.{metric_name} {order}"]
        
        runs = self.search_runs(
            filter_string=filter_string,
            order_by=order_by,
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            logger.info(f"Best run: {best_run.info.run_id} with {metric_name}={best_run.data.metrics.get(metric_name)}")
            return best_run
        
        return None
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to include (None for all)
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                
                row = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                    'start_time': run.info.start_time,
                    'status': run.info.status
                }
                
                # Add parameters
                for key, value in run.data.params.items():
                    row[f'param_{key}'] = value
                
                # Add metrics
                for key, value in run.data.metrics.items():
                    if metrics is None or key in metrics:
                        row[f'metric_{key}'] = value
                
                comparison_data.append(row)
            
            except Exception as e:
                logger.warning(f"Failed to get run {run_id}: {e}")
        
        df = pd.DataFrame(comparison_data)
        logger.info(f"Compared {len(comparison_data)} runs")
        return df
    
    def get_run_metrics_history(
        self,
        run_id: str,
        metric_name: str
    ) -> pd.DataFrame:
        """
        Get metric history for a run.
        
        Args:
            run_id: Run ID
            metric_name: Name of metric
            
        Returns:
            DataFrame with metric history
        """
        try:
            metric_history = self.client.get_metric_history(run_id, metric_name)
            
            data = [{
                'step': m.step,
                'value': m.value,
                'timestamp': m.timestamp
            } for m in metric_history]
            
            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} metric history points")
            return df
        
        except Exception as e:
            logger.error(f"Failed to get metric history: {e}")
            return pd.DataFrame()
    
    def delete_run(self, run_id: str):
        """Delete a run"""
        try:
            self.client.delete_run(run_id)
            logger.info(f"Deleted run: {run_id}")
        except Exception as e:
            logger.error(f"Failed to delete run: {e}")
    
    def delete_model_version(self, model_name: str, version: str):
        """Delete a model version"""
        try:
            self.client.delete_model_version(model_name, version)
            logger.info(f"Deleted model version: {model_name} v{version}")
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")


def setup_mlflow_tracking():
    """
    Set up MLflow tracking server configuration.
    
    This function should be called at application startup.
    """
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(Config.MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            mlflow.create_experiment(Config.MLFLOW_EXPERIMENT_NAME)
            logger.info(f"Created MLflow experiment: {Config.MLFLOW_EXPERIMENT_NAME}")
        else:
            logger.info(f"Using existing MLflow experiment: {Config.MLFLOW_EXPERIMENT_NAME}")
    except Exception as e:
        logger.warning(f"Could not set up MLflow experiment: {e}")


def log_training_run(
    model_name: str,
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[Dict[str, str]] = None,
    tags: Optional[Dict[str, str]] = None,
    register: bool = True
) -> Tuple[str, Optional[Any]]:
    """
    Convenience function to log a complete training run.
    
    Args:
        model_name: Name for the model
        model: Trained model
        params: Training parameters
        metrics: Final metrics
        artifacts: Dictionary of artifact paths
        tags: Run tags
        register: Whether to register model
        
    Returns:
        Tuple of (run_id, model_version)
    """
    manager = MLflowManager()
    
    # Start run
    run = manager.start_run(run_name=f"{model_name}_training", tags=tags)
    run_id = run.info.run_id
    
    try:
        # Log parameters
        manager.log_params(params)
        
        # Log metrics
        manager.log_metrics(metrics)
        
        # Log artifacts
        if artifacts:
            for artifact_path in artifacts.values():
                if Path(artifact_path).exists():
                    manager.log_artifact(artifact_path)
        
        # Log model
        manager.log_model(model, artifact_path="model")
        
        # Register model if requested
        model_version = None
        if register:
            model_uri = f"runs:/{run_id}/model"
            model_version = manager.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags,
                description=f"Model trained with {params}"
            )
        
        logger.info(f"Successfully logged training run: {run_id}")
        return run_id, model_version
    
    finally:
        manager.end_run()


def get_production_model(model_name: str) -> Optional[Any]:
    """
    Get the production version of a model.
    
    Args:
        model_name: Name of registered model
        
    Returns:
        Loaded model or None
    """
    manager = MLflowManager()
    return manager.load_model(model_name, stage="Production")


def promote_model_to_production(
    model_name: str,
    version: str,
    archive_existing: bool = True
):
    """
    Promote a model version to production.
    
    Args:
        model_name: Name of registered model
        version: Version to promote
        archive_existing: Whether to archive existing production versions
    """
    manager = MLflowManager()
    manager.transition_model_stage(
        model_name=model_name,
        version=version,
        stage="Production",
        archive_existing=archive_existing
    )
    logger.info(f"Promoted {model_name} v{version} to Production")


if __name__ == '__main__':
    # Example usage
    setup_mlflow_tracking()
    
    manager = MLflowManager()
    
    # Search for best run
    best_run = manager.get_best_run(metric_name='val_loss', ascending=True)
    if best_run:
        print(f"Best run ID: {best_run.info.run_id}")
        print(f"Best val_loss: {best_run.data.metrics.get('val_loss')}")
