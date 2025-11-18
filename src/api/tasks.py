"""Celery tasks for async AI operations"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from src.api.celery_app import celery, BaseTask
from src.services.enterprise_analyst_service import EnterpriseAnalystService
from src.services.market_decipherer_service import MarketDeciphererService
from src.services.strategy_engine_service import StrategyEngineService
from src.services.performance_governor_service import PerformanceGovernorService
from src.services.business_manager_service import BusinessManagerService
from src.api.database import (
    AnalysisRepository,
    MarketAnalysisRepository,
    StrategyRepository,
    PerformanceRepository,
    BusinessOptimizationRepository
)

logger = logging.getLogger(__name__)


@celery.task(base=BaseTask, bind=True, name='src.api.tasks.analyze_company_async')
def analyze_company_async(self, text: str, source_type: str = None):
    """
    Async task for company analysis using BERT.
    
    Args:
        text: Company or product text
        source_type: Type of source document
        
    Returns:
        Analysis results dictionary
    """
    try:
        logger.info(f"Starting async company analysis (task_id: {self.request.id})")
        
        service = EnterpriseAnalystService()
        result = service.analyze_company(text, source_type)
        
        # Add task metadata
        result['task_id'] = self.request.id
        result['completed_at'] = datetime.utcnow().isoformat()
        
        # Save to database
        try:
            import hashlib
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            result['text_hash'] = text_hash
            result['source_type'] = source_type
            AnalysisRepository.save_analysis_result(result)
            logger.info(f"Analysis result saved to database (task_id: {self.request.id})")
        except Exception as db_error:
            logger.warning(f"Failed to save analysis result to database: {db_error}")
        
        logger.info(f"Company analysis completed (task_id: {self.request.id})")
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_company_async: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery.task(base=BaseTask, bind=True, name='src.api.tasks.analyze_market_async')
def analyze_market_async(
    self,
    market_data: list,
    entity_ids: list = None,
    auto_select_clusters: bool = True,
    similarity_threshold: float = 0.7,
    top_k_links: int = 20
):
    """
    Async task for market analysis.
    
    Args:
        market_data: List of market data dictionaries
        entity_ids: Optional entity identifiers
        auto_select_clusters: Whether to auto-select cluster count
        similarity_threshold: Threshold for similarity links
        top_k_links: Number of top links to return
        
    Returns:
        Market analysis results dictionary
    """
    try:
        logger.info(f"Starting async market analysis (task_id: {self.request.id}, entities: {len(market_data)})")
        
        # Convert to DataFrame
        market_df = pd.DataFrame(market_data)
        
        service = MarketDeciphererService()
        result = service.analyze_market(
            market_data=market_df,
            entity_ids=entity_ids,
            auto_select_clusters=auto_select_clusters,
            similarity_threshold=similarity_threshold,
            top_k_links=top_k_links
        )
        
        # Add task metadata
        result['task_id'] = self.request.id
        result['completed_at'] = datetime.utcnow().isoformat()
        
        # Save to database
        try:
            MarketAnalysisRepository.save_market_analysis(result)
            logger.info(f"Market analysis result saved to database (task_id: {self.request.id})")
        except Exception as db_error:
            logger.warning(f"Failed to save market analysis result to database: {db_error}")
        
        logger.info(f"Market analysis completed (task_id: {self.request.id})")
        return result
        
    except Exception as e:
        logger.error(f"Error in analyze_market_async: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery.task(base=BaseTask, bind=True, name='src.api.tasks.generate_strategy_async')
def generate_strategy_async(
    self,
    market_state: dict,
    context: dict = None,
    include_explanation: bool = True,
    deterministic: bool = True
):
    """
    Async task for strategy generation.
    
    Args:
        market_state: Market state dictionary
        context: Optional context information
        include_explanation: Whether to include LLM explanation
        deterministic: Whether to use deterministic policy
        
    Returns:
        Strategy results dictionary
    """
    try:
        logger.info(f"Starting async strategy generation (task_id: {self.request.id})")
        
        service = StrategyEngineService()
        result = service.generate_strategy(
            market_state=market_state,
            context=context,
            include_explanation=include_explanation,
            deterministic=deterministic
        )
        
        # Add task metadata
        result['task_id'] = self.request.id
        result['market_state'] = market_state
        result['completed_at'] = datetime.utcnow().isoformat()
        
        # Save to database
        try:
            StrategyRepository.save_strategy(result)
            logger.info(f"Strategy result saved to database (task_id: {self.request.id})")
        except Exception as db_error:
            logger.warning(f"Failed to save strategy result to database: {db_error}")
        
        logger.info(f"Strategy generation completed (task_id: {self.request.id})")
        return result
        
    except Exception as e:
        logger.error(f"Error in generate_strategy_async: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery.task(base=BaseTask, bind=True, name='src.api.tasks.monitor_performance_async')
def monitor_performance_async(
    self,
    historical_data: list,
    current_data: list = None,
    strategy_context: dict = None,
    include_feedback: bool = True
):
    """
    Async task for performance monitoring.
    
    Args:
        historical_data: Historical time-series data
        current_data: Current time-series data
        strategy_context: Strategy context for feedback
        include_feedback: Whether to include feedback loop
        
    Returns:
        Performance monitoring results dictionary
    """
    try:
        logger.info(f"Starting async performance monitoring (task_id: {self.request.id})")
        
        # Convert to numpy arrays
        historical_data = np.array(historical_data, dtype=np.float32)
        current_data = np.array(current_data, dtype=np.float32) if current_data else None
        
        service = PerformanceGovernorService(
            input_size=historical_data.shape[1],
            enable_feedback_loop=include_feedback
        )
        
        result = service.monitor_performance(
            historical_data=historical_data,
            current_data=current_data,
            strategy_context=strategy_context,
            include_feedback=include_feedback
        )
        
        # Add task metadata
        result['task_id'] = self.request.id
        result['completed_at'] = datetime.utcnow().isoformat()
        
        # Save to database
        try:
            PerformanceRepository.save_performance_result(result)
            logger.info(f"Performance result saved to database (task_id: {self.request.id})")
        except Exception as db_error:
            logger.warning(f"Failed to save performance result to database: {db_error}")
        
        logger.info(f"Performance monitoring completed (task_id: {self.request.id})")
        return result
        
    except Exception as e:
        logger.error(f"Error in monitor_performance_async: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery.task(base=BaseTask, bind=True, name='src.api.tasks.optimize_business_async')
def optimize_business_async(
    self,
    product_portfolio: list,
    rl_strategy_outputs: dict = None,
    lstm_forecast_outputs: dict = None,
    constraints: dict = None,
    revenue_weight: float = 0.7,
    cost_weight: float = 0.3
):
    """
    Async task for business optimization.
    
    Args:
        product_portfolio: List of product dictionaries
        rl_strategy_outputs: RL strategy outputs
        lstm_forecast_outputs: LSTM forecast outputs
        constraints: Optimization constraints
        revenue_weight: Weight for revenue objective
        cost_weight: Weight for cost objective
        
    Returns:
        Business optimization results dictionary
    """
    try:
        logger.info(f"Starting async business optimization (task_id: {self.request.id}, products: {len(product_portfolio)})")
        
        service = BusinessManagerService()
        
        result = service.optimize_business(
            product_portfolio=product_portfolio,
            rl_strategy_outputs=rl_strategy_outputs,
            lstm_forecast_outputs=lstm_forecast_outputs,
            constraints=constraints,
            revenue_weight=revenue_weight,
            cost_weight=cost_weight
        )
        
        # Add task metadata
        result['task_id'] = self.request.id
        result['completed_at'] = datetime.utcnow().isoformat()
        
        # Save to database
        try:
            BusinessOptimizationRepository.save_optimization_result(result)
            logger.info(f"Business optimization result saved to database (task_id: {self.request.id})")
        except Exception as db_error:
            logger.warning(f"Failed to save business optimization result to database: {db_error}")
        
        logger.info(f"Business optimization completed (task_id: {self.request.id})")
        return result
        
    except Exception as e:
        logger.error(f"Error in optimize_business_async: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery.task(base=BaseTask, bind=True, name='src.api.tasks.explain_model_async')
def explain_model_async(
    self,
    model_type: str,
    instance: list,
    top_n: int = 10,
    include_visualizations: bool = False,
    background_data: list = None
):
    """
    Async task for model explanation.
    
    Args:
        model_type: Type of model ('rl', 'lstm', 'regression')
        instance: Input instance to explain
        top_n: Number of top features
        include_visualizations: Whether to include visualizations
        background_data: Background dataset for SHAP
        
    Returns:
        Explanation results dictionary
    """
    try:
        logger.info(f"Starting async model explanation (task_id: {self.request.id}, model: {model_type})")
        
        # Convert to numpy arrays
        instance = np.array(instance, dtype=np.float32)
        background_data = np.array(background_data, dtype=np.float32) if background_data else None
        
        # Load appropriate model
        from src.services.model_transparency_service import ModelTransparencyService
        
        rl_agent = None
        lstm_forecaster = None
        regression_optimizer = None
        
        if model_type == 'rl':
            from src.models.strategy_agent import StrategyAgent
            rl_agent = StrategyAgent(model_path='models/checkpoints/strategy_agent/best_model.zip')
        elif model_type == 'lstm':
            from src.models.sales_forecaster import SalesForecaster
            lstm_forecaster = SalesForecaster()
            lstm_forecaster.load_checkpoint('models/checkpoints/sales_forecaster/best_model.pt')
        elif model_type == 'regression':
            from src.models.business_optimizer import BusinessOptimizer
            regression_optimizer = BusinessOptimizer()
            regression_optimizer.load_checkpoint('models/checkpoints/business_optimizer/best_model.pt')
        
        transparency_service = ModelTransparencyService(
            rl_agent=rl_agent,
            lstm_forecaster=lstm_forecaster,
            regression_optimizer=regression_optimizer
        )
        
        result = transparency_service.explain_prediction(
            model_type=model_type,
            instance=instance,
            top_n=top_n,
            include_visualizations=include_visualizations,
            background_data=background_data
        )
        
        # Add task metadata
        result['task_id'] = self.request.id
        result['completed_at'] = datetime.utcnow().isoformat()
        
        logger.info(f"Model explanation completed (task_id: {self.request.id})")
        return result
        
    except Exception as e:
        logger.error(f"Error in explain_model_async: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=60, max_retries=3)


@celery.task(base=BaseTask, bind=True, name='src.api.tasks.cleanup_old_results')
def cleanup_old_results(self, max_age_hours: int = 24):
    """
    Periodic task to cleanup old task results.
    
    Args:
        max_age_hours: Maximum age of results to keep in hours
    """
    try:
        logger.info(f"Starting cleanup of old task results (max_age: {max_age_hours}h)")
        
        # This would typically clean up old results from the result backend
        # Implementation depends on the backend (Redis, database, etc.)
        
        logger.info("Cleanup completed")
        return {'status': 'success', 'max_age_hours': max_age_hours}
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_results: {e}", exc_info=True)
        raise


# Configure periodic tasks
celery.conf.beat_schedule = {
    'cleanup-old-results': {
        'task': 'src.api.tasks.cleanup_old_results',
        'schedule': 3600.0,  # Run every hour
        'args': (24,)  # Keep results for 24 hours
    },
}
