"""Service layer for Market Decipherer module"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from src.models.market_decipherer import MarketDecipherer
from src.utils.config import Config

logger = logging.getLogger(__name__)


class MarketDeciphererService:
    """
    Service class for managing Market Decipherer operations.
    Handles model initialization, caching, and business logic.
    """
    
    _instance: Optional['MarketDeciphererService'] = None
    _model: Optional[MarketDecipherer] = None
    
    def __new__(cls):
        """Singleton pattern to ensure single model instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the service"""
        if self._model is None:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Market Decipherer model"""
        try:
            config = Config()
            
            # Configure autoencoder
            autoencoder_config = {
                'input_dim': 512,  # Will be adjusted dynamically
                'hidden_dims': [256, 128],
                'latent_dim': 64,
                'dropout': 0.2
            }
            
            self._model = MarketDecipherer(
                autoencoder_config=autoencoder_config,
                clustering_method='kmeans',  # or 'dbscan'
                gnn_type='graphsage',  # or 'gat'
                device=config.DEVICE
            )
            
            logger.info("MarketDecipherer model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MarketDecipherer model: {e}")
            raise
    
    def analyze_market(
        self,
        market_data: pd.DataFrame,
        entity_ids: Optional[List[str]] = None,
        auto_select_clusters: bool = True,
        similarity_threshold: float = 0.7,
        top_k_links: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze market data to identify segments and relationships.
        
        Args:
            market_data: Market data DataFrame with entity features
            entity_ids: Optional list of entity identifiers
            auto_select_clusters: Whether to automatically select cluster count
            similarity_threshold: Threshold for creating graph edges (0-1)
            top_k_links: Number of top relationship predictions to return
            
        Returns:
            Analysis results dictionary containing clusters, graph, and potential clients
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        # Validate input
        if market_data.empty:
            raise ValueError("Market data cannot be empty")
        
        if len(market_data) > 10000:
            logger.warning(f"Large dataset detected: {len(market_data)} entities. Processing may take time.")
        
        # Validate parameters
        if not 0 <= similarity_threshold <= 1:
            logger.warning(f"Invalid similarity_threshold: {similarity_threshold}, using default 0.7")
            similarity_threshold = 0.7
        
        if top_k_links < 1:
            logger.warning(f"Invalid top_k_links: {top_k_links}, using default 20")
            top_k_links = 20
        
        # Perform analysis
        try:
            results = self._model.analyze_market(
                market_data=market_data,
                entity_ids=entity_ids,
                auto_select_clusters=auto_select_clusters,
                similarity_threshold=similarity_threshold,
                top_k_links=top_k_links
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error during market analysis: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if self._model is None:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'ready',
            'clustering_method': self._model.clustering_method,
            'gnn_type': self._model.gnn_type,
            'device': self._model.device,
            'autoencoder_config': self._model.autoencoder_config,
            'components': {
                'autoencoder': self._model.autoencoder is not None,
                'clusterer': self._model.clusterer is not None,
                'gnn': self._model.gnn is not None
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the service.
        
        Returns:
            Health status dictionary
        """
        try:
            if self._model is None:
                return {
                    'status': 'unhealthy',
                    'message': 'Model not initialized'
                }
            
            # Perform a simple test with synthetic data
            import numpy as np
            test_data = pd.DataFrame(
                np.random.randn(50, 10),
                columns=[f'feature_{i}' for i in range(10)]
            )
            
            result = self._model.analyze_market(
                test_data,
                auto_select_clusters=True,
                top_k_links=5
            )
            
            return {
                'status': 'healthy',
                'message': 'Service is operational',
                'test_processing_time_seconds': result.get('processing_time_seconds', 0),
                'test_clusters_found': result['clusters']['n_clusters']
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'message': str(e)
            }
    
    def save_models(self, path: str):
        """
        Save trained models to disk.
        
        Args:
            path: Directory path to save models
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        self._model.save_models(path)
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str):
        """
        Load trained models from disk.
        
        Args:
            path: Directory path to load models from
        """
        if self._model is None:
            self._initialize_model()
        
        self._model.load_models(path)
        logger.info(f"Models loaded from {path}")
