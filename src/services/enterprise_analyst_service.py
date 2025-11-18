"""Service layer for Enterprise Analyst module"""

import logging
from typing import Dict, Any, Optional
from src.models.enterprise_analyst import EnterpriseAnalyst
from src.utils.config import Config

logger = logging.getLogger(__name__)


class EnterpriseAnalystService:
    """
    Service class for managing Enterprise Analyst operations.
    Handles model initialization, caching, and business logic.
    """
    
    _instance: Optional['EnterpriseAnalystService'] = None
    _model: Optional[EnterpriseAnalyst] = None
    
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
        """Initialize the BERT model"""
        try:
            config = Config()
            self._model = EnterpriseAnalyst(
                model_path=config.BERT_MODEL_PATH,
                device=config.DEVICE,
                max_length=config.BERT_MAX_LENGTH,
                use_mlflow=False  # Set to True when MLflow integration is ready
            )
            logger.info("EnterpriseAnalyst model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EnterpriseAnalyst model: {e}")
            raise
    
    def analyze_company(
        self,
        text: str,
        source_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze company or product text.
        
        Args:
            text: Input text to analyze
            source_type: Type of source document
            
        Returns:
            Analysis results dictionary
        """
        if self._model is None:
            raise RuntimeError("Model not initialized")
        
        # Validate source type
        valid_source_types = ['annual_report', 'product_summary', 'whitepaper', None]
        if source_type not in valid_source_types:
            logger.warning(f"Invalid source_type: {source_type}, using None")
            source_type = None
        
        return self._model.analyze_company(text, source_type)
    
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
            'model_path': self._model.model_path,
            'device': self._model.device,
            'max_length': self._model.max_length,
            'categories': self._model.categories,
            'domains': self._model.domains
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
            
            # Perform a simple inference test
            test_text = "This is a test company that provides software solutions."
            result = self._model.analyze_company(test_text)
            
            return {
                'status': 'healthy',
                'message': 'Service is operational',
                'test_inference_time_ms': result.get('processing_time_ms', 0)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'message': str(e)
            }
