"""Health check endpoints and utilities for the Sales Strategist API"""

import logging
import time
from typing import Dict, Any
from flask import Blueprint, jsonify
from src.api.database import postgres_manager, mongodb_manager
from src.services.enterprise_analyst_service import EnterpriseAnalystService
from src.services.market_decipherer_service import MarketDeciphererService
from src.services.strategy_engine_service import StrategyEngineService
from src.services.performance_governor_service import PerformanceGovernorService
from src.services.business_manager_service import BusinessManagerService

logger = logging.getLogger(__name__)

# Create blueprint for health checks
health_bp = Blueprint('health', __name__)


def check_database_health() -> Dict[str, Any]:
    """
    Check health of database connections.
    
    Returns:
        Dictionary with database health status
    """
    postgres_healthy = False
    mongodb_healthy = False
    
    try:
        postgres_healthy = postgres_manager.health_check()
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
    
    try:
        mongodb_healthy = mongodb_manager.health_check()
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
    
    return {
        'postgresql': {
            'status': 'healthy' if postgres_healthy else 'unhealthy',
            'message': 'Connected' if postgres_healthy else 'Connection failed'
        },
        'mongodb': {
            'status': 'healthy' if mongodb_healthy else 'unhealthy',
            'message': 'Connected' if mongodb_healthy else 'Connection failed'
        },
        'overall': 'healthy' if (postgres_healthy and mongodb_healthy) else 'unhealthy'
    }


def check_model_readiness() -> Dict[str, Any]:
    """
    Check if AI models are loaded and ready.
    
    Returns:
        Dictionary with model readiness status
    """
    models_status = {}
    
    # Check Enterprise Analyst (BERT)
    try:
        ea_service = EnterpriseAnalystService()
        ea_health = ea_service.health_check()
        models_status['enterprise_analyst'] = {
            'status': ea_health['status'],
            'ready': ea_health['status'] == 'healthy',
            'message': ea_health.get('message', 'Model loaded')
        }
    except Exception as e:
        logger.error(f"Enterprise Analyst health check failed: {e}")
        models_status['enterprise_analyst'] = {
            'status': 'unhealthy',
            'ready': False,
            'message': str(e)
        }
    
    # Check Market Decipherer (Autoencoder + GNN)
    try:
        md_service = MarketDeciphererService()
        md_health = md_service.health_check()
        models_status['market_decipherer'] = {
            'status': md_health['status'],
            'ready': md_health['status'] == 'healthy',
            'message': md_health.get('message', 'Models loaded')
        }
    except Exception as e:
        logger.error(f"Market Decipherer health check failed: {e}")
        models_status['market_decipherer'] = {
            'status': 'unhealthy',
            'ready': False,
            'message': str(e)
        }
    
    # Check Strategy Engine (RL + LLM)
    try:
        se_service = StrategyEngineService()
        models_status['strategy_engine'] = {
            'status': 'healthy',
            'ready': se_service.agent.model is not None,
            'message': 'RL agent loaded' if se_service.agent.model else 'RL agent not loaded',
            'llm_available': se_service.llm_available
        }
    except Exception as e:
        logger.error(f"Strategy Engine health check failed: {e}")
        models_status['strategy_engine'] = {
            'status': 'unhealthy',
            'ready': False,
            'message': str(e)
        }
    
    # Check Performance Governor (LSTM)
    try:
        pg_service = PerformanceGovernorService()
        models_status['performance_governor'] = {
            'status': 'healthy',
            'ready': pg_service.lstm_model is not None,
            'message': 'LSTM model loaded' if pg_service.lstm_model else 'LSTM model not loaded'
        }
    except Exception as e:
        logger.error(f"Performance Governor health check failed: {e}")
        models_status['performance_governor'] = {
            'status': 'unhealthy',
            'ready': False,
            'message': str(e)
        }
    
    # Check Business Manager (Regression)
    try:
        bm_service = BusinessManagerService()
        models_status['business_manager'] = {
            'status': 'healthy',
            'ready': bm_service.regression_model is not None,
            'message': 'Regression model loaded' if bm_service.regression_model else 'Regression model not loaded'
        }
    except Exception as e:
        logger.error(f"Business Manager health check failed: {e}")
        models_status['business_manager'] = {
            'status': 'unhealthy',
            'ready': False,
            'message': str(e)
        }
    
    # Determine overall readiness
    all_ready = all(model.get('ready', False) for model in models_status.values())
    
    return {
        'models': models_status,
        'overall': 'ready' if all_ready else 'not_ready',
        'ready_count': sum(1 for model in models_status.values() if model.get('ready', False)),
        'total_count': len(models_status)
    }


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        200: Service is healthy
        503: Service is unhealthy
    """
    try:
        return jsonify({
            'status': 'healthy',
            'message': 'API service is running',
            'timestamp': time.time()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'message': str(e),
            'timestamp': time.time()
        }), 503


@health_bp.route('/health/ready', methods=['GET'])
def readiness_check():
    """
    Readiness check endpoint for Kubernetes.
    Checks if the service is ready to accept traffic.
    
    Returns:
        200: Service is ready
        503: Service is not ready
    """
    try:
        # Check database connections
        db_health = check_database_health()
        
        # Check model readiness
        model_health = check_model_readiness()
        
        # Service is ready if databases are healthy and at least some models are loaded
        is_ready = (
            db_health['overall'] == 'healthy' and
            model_health['ready_count'] >= 2  # At least 2 models should be ready
        )
        
        status_code = 200 if is_ready else 503
        
        return jsonify({
            'status': 'ready' if is_ready else 'not_ready',
            'message': 'Service is ready to accept traffic' if is_ready else 'Service is not ready',
            'databases': db_health,
            'models': {
                'ready_count': model_health['ready_count'],
                'total_count': model_health['total_count'],
                'overall': model_health['overall']
            },
            'timestamp': time.time()
        }), status_code
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            'status': 'not_ready',
            'message': str(e),
            'timestamp': time.time()
        }), 503


@health_bp.route('/health/live', methods=['GET'])
def liveness_check():
    """
    Liveness check endpoint for Kubernetes.
    Checks if the service is alive and should not be restarted.
    
    Returns:
        200: Service is alive
        503: Service is dead (should be restarted)
    """
    try:
        # Basic check - if we can respond, we're alive
        return jsonify({
            'status': 'alive',
            'message': 'Service is alive',
            'timestamp': time.time()
        }), 200
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return jsonify({
            'status': 'dead',
            'message': str(e),
            'timestamp': time.time()
        }), 503


@health_bp.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """
    Detailed health check endpoint with comprehensive status information.
    
    Returns:
        200: Service is healthy
        503: Service has issues
    """
    try:
        start_time = time.time()
        
        # Check database health
        db_health = check_database_health()
        
        # Check model readiness
        model_health = check_model_readiness()
        
        # Determine overall health
        overall_healthy = (
            db_health['overall'] == 'healthy' and
            model_health['overall'] == 'ready'
        )
        
        processing_time = time.time() - start_time
        
        status_code = 200 if overall_healthy else 503
        
        return jsonify({
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'message': 'All systems operational' if overall_healthy else 'Some systems have issues',
            'databases': db_health,
            'models': model_health,
            'processing_time_seconds': round(processing_time, 3),
            'timestamp': time.time()
        }), status_code
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'message': str(e),
            'timestamp': time.time()
        }), 503


@health_bp.route('/health/startup', methods=['GET'])
def startup_check():
    """
    Startup check endpoint for Kubernetes.
    Checks if the service has completed initialization.
    
    Returns:
        200: Service has started successfully
        503: Service is still starting
    """
    try:
        # Check if at least one model is loaded
        model_health = check_model_readiness()
        
        is_started = model_health['ready_count'] >= 1
        
        status_code = 200 if is_started else 503
        
        return jsonify({
            'status': 'started' if is_started else 'starting',
            'message': 'Service has started' if is_started else 'Service is still initializing',
            'models_ready': model_health['ready_count'],
            'models_total': model_health['total_count'],
            'timestamp': time.time()
        }), status_code
        
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        return jsonify({
            'status': 'starting',
            'message': str(e),
            'timestamp': time.time()
        }), 503
