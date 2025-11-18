"""
Model Transparency Service

This service integrates SHAP explainers for all model types and provides
a unified interface for model explanations with caching.
"""

import os
from typing import Dict, Any, Optional, List
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta

from src.models.shap_explainer import SHAPExplainer
from src.models.strategy_agent import StrategyAgent
from src.models.sales_forecaster import SalesForecaster
from src.models.business_optimizer import BusinessOptimizer


class ExplanationCache:
    """Simple in-memory cache for explanations."""
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache = {}
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached explanation if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < self.ttl:
                return entry['data']
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def set(self, key: str, data: Dict[str, Any]):
        """Store explanation in cache."""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()



class ModelTransparencyService:
    """
    Service for providing model explanations across all AI modules.
    Integrates SHAP explainers with caching for performance.
    """
    
    def __init__(
        self,
        rl_agent: Optional[StrategyAgent] = None,
        lstm_forecaster: Optional[SalesForecaster] = None,
        regression_optimizer: Optional[BusinessOptimizer] = None,
        cache_ttl: int = 3600,
        device: str = 'cpu'
    ):
        """
        Initialize model transparency service.
        
        Args:
            rl_agent: RL strategy agent
            lstm_forecaster: LSTM sales forecaster
            regression_optimizer: Regression business optimizer
            cache_ttl: Cache time-to-live in seconds
            device: Device for computations
        """
        self.rl_agent = rl_agent
        self.lstm_forecaster = lstm_forecaster
        self.regression_optimizer = regression_optimizer
        self.device = device
        
        # Initialize cache
        self.cache = ExplanationCache(ttl_seconds=cache_ttl)
        
        # Initialize explainers (lazy initialization)
        self.explainers = {}
        
        # Feature names for different models
        self.feature_names = {
            'rl': self._get_rl_feature_names(),
            'lstm': self._get_lstm_feature_names(),
            'regression': self._get_regression_feature_names()
        }
    
    def _get_rl_feature_names(self) -> List[str]:
        """Get feature names for RL agent."""
        if self.rl_agent is None:
            return []
        
        num_competitors = self.rl_agent.num_competitors
        names = ['market_demand']
        names.extend([f'competitor_price_{i+1}' for i in range(num_competitors)])
        names.extend(['sales_volume', 'conversion_rate', 'inventory_level', 'market_trend'])
        return names
    
    def _get_lstm_feature_names(self) -> List[str]:
        """Get feature names for LSTM forecaster."""
        if self.lstm_forecaster is None:
            return []
        
        # Generic feature names for time-series
        return [f'feature_{i+1}' for i in range(self.lstm_forecaster.input_size)]
    
    def _get_regression_feature_names(self) -> List[str]:
        """Get feature names for regression optimizer."""
        if self.regression_optimizer is None:
            return []
        
        # Generic feature names
        return [f'feature_{i+1}' for i in range(self.regression_optimizer.input_size)]
    
    def _get_or_create_explainer(
        self,
        model_type: str,
        background_data: Optional[np.ndarray] = None
    ) -> SHAPExplainer:
        """Get or create SHAP explainer for model type."""
        if model_type in self.explainers:
            return self.explainers[model_type]
        
        # Select model and feature names
        if model_type == 'rl':
            if self.rl_agent is None:
                raise ValueError("RL agent not initialized")
            model = self.rl_agent
            feature_names = self.feature_names['rl']
        elif model_type == 'lstm':
            if self.lstm_forecaster is None:
                raise ValueError("LSTM forecaster not initialized")
            model = self.lstm_forecaster
            feature_names = self.feature_names['lstm']
        elif model_type == 'regression':
            if self.regression_optimizer is None:
                raise ValueError("Regression optimizer not initialized")
            model = self.regression_optimizer
            feature_names = self.feature_names['regression']
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create explainer
        explainer = SHAPExplainer(
            model=model,
            model_type=model_type,
            feature_names=feature_names,
            background_data=background_data,
            device=self.device
        )
        
        self.explainers[model_type] = explainer
        return explainer
    
    def explain_prediction(
        self,
        model_type: str,
        instance: np.ndarray,
        top_n: int = 10,
        include_visualizations: bool = False,
        background_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Explain a single prediction with top-N influential features.
        
        Args:
            model_type: Type of model ('rl', 'lstm', 'regression')
            instance: Input instance to explain
            top_n: Number of top features to return
            include_visualizations: Whether to include visualization images
            background_data: Background data for SHAP (optional)
            
        Returns:
            Explanation with top-N features and optional visualizations
        """
        # Check cache
        cache_key = self.cache._generate_key({
            'model_type': model_type,
            'instance': instance.tolist(),
            'top_n': top_n,
            'visualizations': include_visualizations
        })
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Get or create explainer
        explainer = self._get_or_create_explainer(model_type, background_data)
        
        # Generate local explanation
        local_explanation = explainer.explain_local(instance)
        
        # Extract top-N features
        top_features = local_explanation['contributions'][:top_n]
        
        # Calculate feature contribution values
        total_contribution = sum(abs(contrib['shap_value']) for contrib in local_explanation['contributions'])
        
        for feature in top_features:
            if total_contribution > 0:
                feature['contribution_pct'] = abs(feature['shap_value']) / total_contribution * 100
            else:
                feature['contribution_pct'] = 0.0
        
        result = {
            'model_type': model_type,
            'base_value': local_explanation['base_value'],
            'top_features': top_features,
            'total_features': len(local_explanation['contributions']),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add visualizations if requested
        if include_visualizations:
            visualizations = explainer.generate_all_visualizations(
                instance,
                global_data=background_data
            )
            result['visualizations'] = visualizations
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
    
    def explain_global(
        self,
        model_type: str,
        data: np.ndarray,
        top_n: int = 10,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate global explanation showing overall feature importance.
        
        Args:
            model_type: Type of model ('rl', 'lstm', 'regression')
            data: Dataset to analyze
            top_n: Number of top features to return
            max_samples: Maximum samples to use for explanation
            
        Returns:
            Global explanation with feature importance
        """
        # Check cache
        cache_key = self.cache._generate_key({
            'model_type': model_type,
            'data_shape': data.shape,
            'data_hash': hashlib.md5(data.tobytes()).hexdigest()[:16],
            'top_n': top_n,
            'max_samples': max_samples
        })
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Get or create explainer
        explainer = self._get_or_create_explainer(model_type, data[:10] if len(data) > 10 else data)
        
        # Generate global explanation
        global_explanation = explainer.explain_global(data, max_samples=max_samples)
        
        # Extract top-N features
        top_features = global_explanation['feature_importance'][:top_n]
        
        # Calculate contribution percentages
        total_importance = sum(f['importance'] for f in global_explanation['feature_importance'])
        
        for feature in top_features:
            if total_importance > 0:
                feature['importance_pct'] = feature['importance'] / total_importance * 100
            else:
                feature['importance_pct'] = 0.0
        
        result = {
            'model_type': model_type,
            'top_features': top_features,
            'total_features': len(global_explanation['feature_importance']),
            'num_samples_analyzed': global_explanation['num_samples'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
    
    def batch_explain(
        self,
        model_type: str,
        instances: List[np.ndarray],
        top_n: int = 10,
        background_data: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple predictions in batch.
        
        Args:
            model_type: Type of model ('rl', 'lstm', 'regression')
            instances: List of instances to explain
            top_n: Number of top features per instance
            background_data: Background data for SHAP (optional)
            
        Returns:
            List of explanations
        """
        explanations = []
        
        for instance in instances:
            try:
                explanation = self.explain_prediction(
                    model_type=model_type,
                    instance=instance,
                    top_n=top_n,
                    include_visualizations=False,
                    background_data=background_data
                )
                explanations.append(explanation)
            except Exception as e:
                explanations.append({
                    'error': str(e),
                    'model_type': model_type
                })
        
        return explanations
    
    def compare_predictions(
        self,
        model_type: str,
        instance1: np.ndarray,
        instance2: np.ndarray,
        top_n: int = 10,
        background_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compare explanations for two different predictions.
        
        Args:
            model_type: Type of model
            instance1: First instance
            instance2: Second instance
            top_n: Number of top features
            background_data: Background data for SHAP
            
        Returns:
            Comparison of explanations
        """
        exp1 = self.explain_prediction(model_type, instance1, top_n, False, background_data)
        exp2 = self.explain_prediction(model_type, instance2, top_n, False, background_data)
        
        # Find common features and differences
        features1 = {f['feature']: f['shap_value'] for f in exp1['top_features']}
        features2 = {f['feature']: f['shap_value'] for f in exp2['top_features']}
        
        common_features = set(features1.keys()) & set(features2.keys())
        
        differences = []
        for feature in common_features:
            diff = features2[feature] - features1[feature]
            differences.append({
                'feature': feature,
                'instance1_shap': features1[feature],
                'instance2_shap': features2[feature],
                'difference': diff
            })
        
        # Sort by absolute difference
        differences.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        return {
            'model_type': model_type,
            'instance1_explanation': exp1,
            'instance2_explanation': exp2,
            'feature_differences': differences,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_feature_contributions(
        self,
        model_type: str,
        instance: np.ndarray,
        background_data: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Get feature contribution values for all features.
        
        Args:
            model_type: Type of model
            instance: Instance to explain
            background_data: Background data for SHAP
            
        Returns:
            Dictionary mapping feature names to contribution values
        """
        explainer = self._get_or_create_explainer(model_type, background_data)
        local_explanation = explainer.explain_local(instance)
        
        contributions = {
            contrib['feature']: contrib['shap_value']
            for contrib in local_explanation['contributions']
        }
        
        return contributions
    
    def clear_cache(self):
        """Clear the explanation cache."""
        self.cache.clear()
    
    def set_models(
        self,
        rl_agent: Optional[StrategyAgent] = None,
        lstm_forecaster: Optional[SalesForecaster] = None,
        regression_optimizer: Optional[BusinessOptimizer] = None
    ):
        """
        Update models and reset explainers.
        
        Args:
            rl_agent: New RL agent
            lstm_forecaster: New LSTM forecaster
            regression_optimizer: New regression optimizer
        """
        if rl_agent is not None:
            self.rl_agent = rl_agent
            self.feature_names['rl'] = self._get_rl_feature_names()
            if 'rl' in self.explainers:
                del self.explainers['rl']
        
        if lstm_forecaster is not None:
            self.lstm_forecaster = lstm_forecaster
            self.feature_names['lstm'] = self._get_lstm_feature_names()
            if 'lstm' in self.explainers:
                del self.explainers['lstm']
        
        if regression_optimizer is not None:
            self.regression_optimizer = regression_optimizer
            self.feature_names['regression'] = self._get_regression_feature_names()
            if 'regression' in self.explainers:
                del self.explainers['regression']
        
        # Clear cache when models change
        self.clear_cache()
