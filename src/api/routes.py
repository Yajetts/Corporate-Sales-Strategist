"""API routes for the Sales Strategist System"""

import logging
from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from src.api.schemas import (
    AnalyzeCompanyRequestSchema,
    AnalyzeCompanyResponseSchema,
    MarketAnalysisRequestSchema,
    MarketAnalysisResponseSchema,
    StrategyRequestSchema,
    StrategyResponseSchema,
    BusinessOptimizerRequestSchema,
    BusinessOptimizerResponseSchema,
    ErrorResponseSchema
)
from src.services.enterprise_analyst_service import EnterpriseAnalystService
from src.services.market_decipherer_service import MarketDeciphererService
from src.services.strategy_engine_service import StrategyEngineService
from src.services.performance_governor_service import PerformanceGovernorService
from src.services.business_manager_service import BusinessManagerService
import pandas as pd
import numpy as np
import time

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize schemas
analyze_company_request_schema = AnalyzeCompanyRequestSchema()
analyze_company_response_schema = AnalyzeCompanyResponseSchema()
market_analysis_request_schema = MarketAnalysisRequestSchema()
market_analysis_response_schema = MarketAnalysisResponseSchema()
strategy_request_schema = StrategyRequestSchema()
strategy_response_schema = StrategyResponseSchema()
business_optimizer_request_schema = BusinessOptimizerRequestSchema()
business_optimizer_response_schema = BusinessOptimizerResponseSchema()
error_response_schema = ErrorResponseSchema()

# Rate limiting for LLM API calls
_strategy_call_times = []
_max_calls_per_minute = 20


def sanitize_input(text: str) -> str:
    """
    Sanitize input text to prevent injection attacks.
    
    Args:
        text: Raw input text
        
    Returns:
        Sanitized text
    """
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


@api_bp.route('/analyze_company', methods=['POST'])
def analyze_company():
    """
    Analyze company or product text using BERT model.
    
    Request Body:
        {
            "text": "string (required)",
            "source_type": "annual_report | product_summary | whitepaper (optional)"
        }
    
    Response:
        {
            "product_category": "string",
            "business_domain": "string",
            "value_proposition": "string",
            "key_features": ["string"],
            "confidence_scores": {
                "category": float,
                "domain": float
            },
            "processing_time_ms": int,
            "source_type": "string"
        }
    
    Status Codes:
        200: Success
        400: Bad Request (validation error)
        500: Internal Server Error
    """
    try:
        # Get request data with error handling for JSON parsing
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.warning(f"JSON parsing error: {json_error}")
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid JSON or missing Content-Type header',
                'status_code': 400
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON',
                'status_code': 400
            }), 400
        
        # Validate request
        try:
            validated_data = analyze_company_request_schema.load(data)
        except ValidationError as err:
            logger.warning(f"Validation error: {err.messages}")
            return jsonify({
                'error': 'Validation Error',
                'message': err.messages,
                'status_code': 400
            }), 400
        
        # Sanitize input
        text = sanitize_input(validated_data['text'])
        source_type = validated_data.get('source_type')
        
        logger.info(f"Processing analyze_company request (text_length: {len(text)}, source_type: {source_type})")
        
        # Get service and analyze
        service = EnterpriseAnalystService()
        result = service.analyze_company(text, source_type)
        
        # Validate response
        validated_response = analyze_company_response_schema.dump(result)
        
        return jsonify(validated_response), 200
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({
            'error': 'Invalid Input',
            'message': str(e),
            'status_code': 400
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return jsonify({
            'error': 'Service Error',
            'message': str(e),
            'status_code': 500
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_company: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for the API service.
    
    Response:
        {
            "status": "healthy | unhealthy",
            "message": "string",
            "services": {
                "enterprise_analyst": {
                    "status": "healthy | unhealthy",
                    "message": "string"
                },
                "market_decipherer": {
                    "status": "healthy | unhealthy",
                    "message": "string"
                },
                "strategy_engine": {
                    "status": "healthy | unhealthy",
                    "message": "string"
                }
            }
        }
    """
    try:
        # Check Enterprise Analyst service
        ea_service = EnterpriseAnalystService()
        ea_health = ea_service.health_check()
        
        # Check Market Decipherer service
        md_service = MarketDeciphererService()
        md_health = md_service.health_check()
        
        # Check Strategy Engine service
        try:
            se_service = StrategyEngineService()
            se_health = {
                'status': 'healthy',
                'message': 'Strategy Engine service is operational'
            }
        except Exception as se_error:
            se_health = {
                'status': 'unhealthy',
                'message': f'Strategy Engine initialization failed: {str(se_error)}'
            }
        
        # Determine overall health
        all_healthy = (
            ea_health['status'] == 'healthy' and
            md_health['status'] == 'healthy' and
            se_health['status'] == 'healthy'
        )
        overall_status = 'healthy' if all_healthy else 'unhealthy'
        
        return jsonify({
            'status': overall_status,
            'message': 'API service health check',
            'services': {
                'enterprise_analyst': ea_health,
                'market_decipherer': md_health,
                'strategy_engine': se_health
            }
        }), 200 if overall_status == 'healthy' else 503
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'message': str(e),
            'services': {}
        }), 503


@api_bp.route('/market_analysis', methods=['POST'])
def market_analysis():
    """
    Analyze market data to identify customer segments and relationships.
    
    Request Body:
        {
            "market_data": [{"feature1": value, "feature2": value, ...}] (required),
            "entity_ids": ["id1", "id2", ...] (optional),
            "auto_select_clusters": bool (optional, default: true),
            "similarity_threshold": float (optional, default: 0.7, range: 0-1),
            "top_k_links": int (optional, default: 20, range: 1-100)
        }
    
    Response:
        {
            "clusters": {
                "labels": [int],
                "n_clusters": int,
                "method": "string",
                "metrics": {},
                "profiles": {}
            },
            "graph": {
                "num_nodes": int,
                "num_edges": int,
                "predicted_links": [{"source": int, "target": int, "score": float}]
            },
            "potential_clients": [{"entity_id": str, "cluster_id": int, "connectivity_score": float}],
            "latent_dimensions": int,
            "processing_time_seconds": float,
            "num_entities": int,
            "task_id": "string" (if async)
        }
    
    Status Codes:
        200: Success
        202: Accepted (async processing)
        400: Bad Request
        500: Internal Server Error
    """
    try:
        # Get request data
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.warning(f"JSON parsing error: {json_error}")
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid JSON or missing Content-Type header',
                'status_code': 400
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON',
                'status_code': 400
            }), 400
        
        # Validate request
        try:
            validated_data = market_analysis_request_schema.load(data)
        except ValidationError as err:
            logger.warning(f"Validation error: {err.messages}")
            return jsonify({
                'error': 'Validation Error',
                'message': err.messages,
                'status_code': 400
            }), 400
        
        # Extract parameters
        market_data_list = validated_data['market_data']
        entity_ids = validated_data.get('entity_ids')
        auto_select_clusters = validated_data.get('auto_select_clusters', True)
        similarity_threshold = validated_data.get('similarity_threshold', 0.7)
        top_k_links = validated_data.get('top_k_links', 20)
        
        # Validate entity_ids length if provided
        if entity_ids and len(entity_ids) != len(market_data_list):
            return jsonify({
                'error': 'Validation Error',
                'message': f'entity_ids length ({len(entity_ids)}) must match market_data length ({len(market_data_list)})',
                'status_code': 400
            }), 400
        
        # Convert to DataFrame
        try:
            market_df = pd.DataFrame(market_data_list)
        except Exception as df_error:
            logger.error(f"Error converting to DataFrame: {df_error}")
            return jsonify({
                'error': 'Invalid Data Format',
                'message': 'Could not convert market_data to DataFrame. Ensure all items have consistent structure.',
                'status_code': 400
            }), 400
        
        logger.info(f"Processing market_analysis request (entities: {len(market_df)}, features: {len(market_df.columns)})")
        
        # Check if async processing is needed (>10,000 entities or processing time > 30s expected)
        if len(market_df) > 5000:
            # For large datasets, recommend async processing
            # For now, we'll process synchronously but add timeout handling
            logger.warning(f"Large dataset detected: {len(market_df)} entities. Consider implementing async processing.")
        
        # Get service and analyze
        service = MarketDeciphererService()
        
        try:
            result = service.analyze_market(
                market_data=market_df,
                entity_ids=entity_ids,
                auto_select_clusters=auto_select_clusters,
                similarity_threshold=similarity_threshold,
                top_k_links=top_k_links
            )
            
            # Validate response
            validated_response = market_analysis_response_schema.dump(result)
            
            return jsonify(validated_response), 200
            
        except TimeoutError:
            return jsonify({
                'error': 'Timeout',
                'message': 'Processing exceeded time limit. Please try with a smaller dataset or use async processing.',
                'status_code': 504
            }), 504
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({
            'error': 'Invalid Input',
            'message': str(e),
            'status_code': 400
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return jsonify({
            'error': 'Service Error',
            'message': str(e),
            'status_code': 500
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in market_analysis: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500


@api_bp.route('/model_info', methods=['GET'])
def model_info():
    """
    Get information about loaded models.
    
    Response:
        {
            "enterprise_analyst": {
                "status": "ready | not_initialized",
                "model_path": "string",
                "device": "string",
                ...
            },
            "market_decipherer": {
                "status": "ready | not_initialized",
                "clustering_method": "string",
                "gnn_type": "string",
                ...
            },
            "strategy_engine": {
                "status": "ready | not_initialized",
                "agent_model_loaded": bool,
                "llm_available": bool,
                ...
            }
        }
    """
    try:
        ea_service = EnterpriseAnalystService()
        ea_info = ea_service.get_model_info()
        
        md_service = MarketDeciphererService()
        md_info = md_service.get_model_info()
        
        try:
            se_service = StrategyEngineService()
            se_info = {
                'status': 'ready' if se_service.agent.model is not None else 'not_initialized',
                'agent_model_loaded': se_service.agent.model is not None,
                'llm_available': se_service.llm_available,
                'caching_enabled': se_service.enable_caching
            }
        except Exception as se_error:
            se_info = {
                'status': 'error',
                'message': str(se_error)
            }
        
        return jsonify({
            'enterprise_analyst': ea_info,
            'market_decipherer': md_info,
            'strategy_engine': se_info
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e)
        }), 500



def check_rate_limit() -> bool:
    """
    Check if rate limit for LLM API calls is exceeded.
    
    Returns:
        True if within rate limit, False otherwise
    """
    global _strategy_call_times
    
    current_time = time.time()
    # Remove calls older than 1 minute
    _strategy_call_times = [t for t in _strategy_call_times if current_time - t < 60]
    
    # Check if limit exceeded
    if len(_strategy_call_times) >= _max_calls_per_minute:
        return False
    
    # Add current call
    _strategy_call_times.append(current_time)
    return True


@api_bp.route('/strategy', methods=['POST'])
def generate_strategy():
    """
    Generate sales and pricing strategy using RL agent and LLM.
    
    Request Body:
        {
            "market_state": {
                "market_demand": float (0-1, required),
                "competitor_prices": [float] (0-1 normalized, required),
                "sales_volume": float (0-1, required),
                "conversion_rate": float (0-1, required),
                "inventory_level": float (0-1, required),
                "market_trend": float (-1 to 1, required)
            },
            "context": {
                "company_name": "string",
                "product_name": "string",
                ...
            } (optional),
            "include_explanation": bool (optional, default: true),
            "deterministic": bool (optional, default: true)
        }
    
    Response:
        {
            "timestamp": "string (ISO format)",
            "market_state": {...},
            "recommendations": {
                "price_adjustment_pct": float,
                "sales_approach": "aggressive | moderate | conservative",
                "promotion_intensity": float (0-1)
            },
            "confidence_score": float (0-1),
            "confidence_level": "high | medium | low",
            "explanation": {
                "summary": "string",
                "rationale": "string",
                "expected_outcomes": "string",
                "risks": "string"
            } (optional),
            "actionable_insights": ["string"],
            "task_id": "string" (if async)
        }
    
    Status Codes:
        200: Success
        202: Accepted (async processing)
        400: Bad Request
        429: Too Many Requests (rate limit exceeded)
        500: Internal Server Error
    """
    try:
        # Check rate limit
        if not check_rate_limit():
            logger.warning("Rate limit exceeded for strategy endpoint")
            return jsonify({
                'error': 'Rate Limit Exceeded',
                'message': f'Maximum {_max_calls_per_minute} requests per minute allowed',
                'status_code': 429,
                'retry_after': 60
            }), 429
        
        # Get request data
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.warning(f"JSON parsing error: {json_error}")
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid JSON or missing Content-Type header',
                'status_code': 400
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON',
                'status_code': 400
            }), 400
        
        # Validate request
        try:
            validated_data = strategy_request_schema.load(data)
        except ValidationError as err:
            logger.warning(f"Validation error: {err.messages}")
            return jsonify({
                'error': 'Validation Error',
                'message': err.messages,
                'status_code': 400
            }), 400
        
        # Extract parameters
        market_state = validated_data['market_state']
        context = validated_data.get('context')
        include_explanation = validated_data.get('include_explanation', True)
        deterministic = validated_data.get('deterministic', True)
        
        logger.info(f"Processing strategy request (market_demand: {market_state['market_demand']}, "
                   f"include_explanation: {include_explanation})")
        
        # Get service and generate strategy
        service = StrategyEngineService()
        
        try:
            result = service.generate_strategy(
                market_state=market_state,
                context=context,
                include_explanation=include_explanation,
                deterministic=deterministic
            )
            
            # Validate response
            validated_response = strategy_response_schema.dump(result)
            
            return jsonify(validated_response), 200
            
        except TimeoutError:
            return jsonify({
                'error': 'Timeout',
                'message': 'Strategy generation exceeded time limit. Please try again.',
                'status_code': 504
            }), 504
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({
            'error': 'Invalid Input',
            'message': str(e),
            'status_code': 400
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return jsonify({
            'error': 'Service Error',
            'message': str(e),
            'status_code': 500
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_strategy: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500


@api_bp.route('/strategy/compare', methods=['POST'])
def compare_strategies():
    """
    Compare strategies across multiple market scenarios.
    
    Request Body:
        {
            "scenarios": [
                {
                    "name": "string (optional)",
                    "market_state": {...}
                }
            ]
        }
    
    Response:
        {
            "scenarios": [
                {
                    "name": "string",
                    "market_state": {...},
                    "recommendations": {...},
                    "confidence_score": float
                }
            ],
            "summary": {
                "price_range": {"min": float, "max": float, "avg": float},
                "most_common_approach": "string",
                "avg_confidence": float,
                "highest_confidence_scenario": "string"
            }
        }
    
    Status Codes:
        200: Success
        400: Bad Request
        429: Too Many Requests
        500: Internal Server Error
    """
    try:
        # Check rate limit
        if not check_rate_limit():
            logger.warning("Rate limit exceeded for strategy comparison endpoint")
            return jsonify({
                'error': 'Rate Limit Exceeded',
                'message': f'Maximum {_max_calls_per_minute} requests per minute allowed',
                'status_code': 429,
                'retry_after': 60
            }), 429
        
        # Get request data
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.warning(f"JSON parsing error: {json_error}")
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid JSON or missing Content-Type header',
                'status_code': 400
            }), 400
        
        if not data or 'scenarios' not in data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must contain "scenarios" field',
                'status_code': 400
            }), 400
        
        scenarios = data['scenarios']
        
        if not isinstance(scenarios, list) or len(scenarios) == 0:
            return jsonify({
                'error': 'Bad Request',
                'message': 'scenarios must be a non-empty list',
                'status_code': 400
            }), 400
        
        if len(scenarios) > 10:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Maximum 10 scenarios allowed per comparison',
                'status_code': 400
            }), 400
        
        # Extract market states and names
        market_states = []
        scenario_names = []
        
        for i, scenario in enumerate(scenarios):
            if 'market_state' not in scenario:
                return jsonify({
                    'error': 'Bad Request',
                    'message': f'Scenario {i} missing market_state field',
                    'status_code': 400
                }), 400
            
            # Validate market state
            try:
                validated_scenario = strategy_request_schema.load({
                    'market_state': scenario['market_state']
                })
                market_states.append(validated_scenario['market_state'])
                scenario_names.append(scenario.get('name', f'Scenario {i+1}'))
            except ValidationError as err:
                return jsonify({
                    'error': 'Validation Error',
                    'message': f'Scenario {i} validation failed: {err.messages}',
                    'status_code': 400
                }), 400
        
        logger.info(f"Processing strategy comparison for {len(scenarios)} scenarios")
        
        # Get service and compare strategies
        service = StrategyEngineService()
        
        result = service.compare_strategies(
            market_states=market_states,
            scenario_names=scenario_names
        )
        
        return jsonify(result), 200
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({
            'error': 'Invalid Input',
            'message': str(e),
            'status_code': 400
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error in compare_strategies: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500


@api_bp.route('/performance', methods=['POST'])
def monitor_performance():
    """
    Monitor performance with LSTM forecasting and anomaly detection.
    
    Request Body:
        {
            "historical_data": [[float]] (required, shape: [n_timesteps, n_features]),
            "current_data": [[float]] (optional, shape: [n_timesteps, n_features]),
            "strategy_context": {
                "price_adjustment_pct": float,
                "sales_approach": "string",
                "promotion_intensity": float
            } (optional),
            "include_feedback": bool (optional, default: true)
        }
    
    Response:
        {
            "timestamp": "string (ISO format)",
            "forecast": {
                "values": [float],
                "horizon_days": int,
                "confidence_interval": {
                    "lower": [float],
                    "upper": [float],
                    "confidence_level": float
                }
            },
            "alerts": {
                "total": int,
                "by_severity": {
                    "low": int,
                    "medium": int,
                    "high": int,
                    "critical": int
                },
                "details": [
                    {
                        "alert_id": "string",
                        "timestamp": "string",
                        "anomaly_type": "string",
                        "severity": "string",
                        "metric_name": "string",
                        "actual_value": float,
                        "expected_value": float,
                        "deviation_pct": float,
                        "description": "string",
                        "recommended_actions": ["string"]
                    }
                ]
            },
            "alert_summary": {
                "total_alerts": int,
                "by_severity": {...},
                "by_type": {...},
                "most_recent": {...},
                "time_window_hours": int
            },
            "feedback": {
                "summary": {...},
                "strategy_weights": {...},
                "rl_weight_adjustments": {...}
            } (optional),
            "trend_analysis": {
                "historical_trend": float,
                "forecast_trend": float,
                "trend_outlook": "string",
                "forecast_vs_current_pct": float
            },
            "processing_time_seconds": float
        }
    
    Status Codes:
        200: Success
        400: Bad Request
        500: Internal Server Error
    """
    try:
        # Get request data
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.warning(f"JSON parsing error: {json_error}")
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid JSON or missing Content-Type header',
                'status_code': 400
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON',
                'status_code': 400
            }), 400
        
        # Validate required fields
        if 'historical_data' not in data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'historical_data field is required',
                'status_code': 400
            }), 400
        
        # Convert to numpy arrays
        try:
            historical_data = np.array(data['historical_data'], dtype=np.float32)
            
            if historical_data.ndim != 2:
                return jsonify({
                    'error': 'Bad Request',
                    'message': 'historical_data must be a 2D array [n_timesteps, n_features]',
                    'status_code': 400
                }), 400
            
            current_data = None
            if 'current_data' in data and data['current_data']:
                current_data = np.array(data['current_data'], dtype=np.float32)
                
                if current_data.ndim != 2:
                    return jsonify({
                        'error': 'Bad Request',
                        'message': 'current_data must be a 2D array [n_timesteps, n_features]',
                        'status_code': 400
                    }), 400
                
                if current_data.shape[1] != historical_data.shape[1]:
                    return jsonify({
                        'error': 'Bad Request',
                        'message': f'current_data features ({current_data.shape[1]}) must match historical_data features ({historical_data.shape[1]})',
                        'status_code': 400
                    }), 400
        
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'Bad Request',
                'message': f'Invalid data format: {str(e)}',
                'status_code': 400
            }), 400
        
        # Extract optional parameters
        strategy_context = data.get('strategy_context')
        include_feedback = data.get('include_feedback', True)
        
        logger.info(f"Processing performance monitoring request (historical: {historical_data.shape}, "
                   f"current: {current_data.shape if current_data is not None else None})")
        
        # Initialize service (with default model path)
        # In production, this should be configured via environment variables
        service = PerformanceGovernorService(
            forecaster_model_path=None,  # Will use default or trained model
            input_size=historical_data.shape[1],
            enable_feedback_loop=include_feedback
        )
        
        # Monitor performance (will use fallback if model not loaded)
        try:
            result = service.monitor_performance(
                historical_data=historical_data,
                current_data=current_data,
                strategy_context=strategy_context,
                include_feedback=include_feedback
            )
            
            return jsonify(result), 200
            
        except TimeoutError:
            return jsonify({
                'error': 'Timeout',
                'message': 'Performance monitoring exceeded time limit. Please try with a smaller dataset.',
                'status_code': 504
            }), 504
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({
            'error': 'Invalid Input',
            'message': str(e),
            'status_code': 400
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return jsonify({
            'error': 'Service Error',
            'message': str(e),
            'status_code': 500
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in monitor_performance: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500


@api_bp.route('/performance/trends', methods=['POST'])
def get_performance_trends():
    """
    Get performance trend analysis.
    
    Request Body:
        {
            "historical_data": [[float]] (required),
            "window_size": int (optional, default: 30)
        }
    
    Response:
        {
            "trend_direction": "increasing | decreasing | stable",
            "trend_coefficient": float,
            "volatility": float,
            "moving_average_7d": float,
            "moving_average_30d": float,
            "current_value": float,
            "period_change_pct": float
        }
    
    Status Codes:
        200: Success
        400: Bad Request
        500: Internal Server Error
    """
    try:
        data = request.get_json()
        
        if not data or 'historical_data' not in data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'historical_data field is required',
                'status_code': 400
            }), 400
        
        historical_data = np.array(data['historical_data'], dtype=np.float32)
        window_size = data.get('window_size', 30)
        
        service = PerformanceGovernorService(input_size=historical_data.shape[1])
        
        result = service.get_performance_trends(
            historical_data=historical_data,
            window_size=window_size
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in get_performance_trends: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500


@api_bp.route('/performance/alerts/critical', methods=['GET'])
def get_critical_alerts():
    """
    Get critical alerts requiring immediate attention.
    
    Query Parameters:
        severity_threshold: "low" | "medium" | "high" | "critical" (optional, default: "high")
    
    Response:
        [
            {
                "alert_id": "string",
                "timestamp": "string",
                "anomaly_type": "string",
                "severity": "string",
                "metric_name": "string",
                "actual_value": float,
                "expected_value": float,
                "deviation_pct": float,
                "description": "string",
                "recommended_actions": ["string"]
            }
        ]
    
    Status Codes:
        200: Success
        400: Bad Request
        500: Internal Server Error
    """
    try:
        from src.models.anomaly_detector import SeverityLevel
        
        severity_param = request.args.get('severity_threshold', 'high').lower()
        
        severity_map = {
            'low': SeverityLevel.LOW,
            'medium': SeverityLevel.MEDIUM,
            'high': SeverityLevel.HIGH,
            'critical': SeverityLevel.CRITICAL
        }
        
        if severity_param not in severity_map:
            return jsonify({
                'error': 'Bad Request',
                'message': f'Invalid severity_threshold. Must be one of: {list(severity_map.keys())}',
                'status_code': 400
            }), 400
        
        severity_threshold = severity_map[severity_param]
        
        service = PerformanceGovernorService()
        
        alerts = service.get_critical_alerts(severity_threshold=severity_threshold)
        
        return jsonify(alerts), 200
        
    except Exception as e:
        logger.error(f"Error in get_critical_alerts: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500


@api_bp.route('/performance/feedback/recommendations', methods=['GET'])
def get_feedback_recommendations():
    """
    Get recommendations based on feedback learning.
    
    Response:
        {
            "enabled": bool,
            "current_weights": {
                "price_sensitivity": float,
                "promotion_effectiveness": float,
                "sales_approach_impact": float,
                "market_responsiveness": float
            },
            "performance_summary": {...},
            "recommendations": [
                {
                    "component": "string",
                    "recommendation": "string",
                    "reason": "string",
                    "priority": "critical | high | medium | low"
                }
            ],
            "recent_adjustments": [...]
        }
    
    Status Codes:
        200: Success
        500: Internal Server Error
    """
    try:
        service = PerformanceGovernorService(enable_feedback_loop=True)
        
        recommendations = service.get_feedback_recommendations()
        
        return jsonify(recommendations), 200
        
    except Exception as e:
        logger.error(f"Error in get_feedback_recommendations: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500



@api_bp.route('/business_optimizer', methods=['POST'])
def optimize_business():
    """
    Optimize manufacturing and resource allocation.
    
    Request Body:
        {
            "product_portfolio": [
                {
                    "name": "string (required)",
                    "sales_history": [float] (optional),
                    "demand_forecast": float (optional),
                    "production_cost": float (optional),
                    "current_inventory": float (optional)
                }
            ] (required),
            "rl_strategy_outputs": {
                "price_adjustments": [float],
                "promotion_intensity": [float]
            } (optional),
            "lstm_forecast_outputs": {
                "sales_forecasts": [float],
                "trend_indicators": [float]
            } (optional),
            "constraints": {
                "min_production": [float],
                "max_production": [float],
                "total_budget": float,
                "capacity_limit": float
            } (optional),
            "revenue_weight": float (0-1, optional, default: 0.7),
            "cost_weight": float (0-1, optional, default: 0.3)
        }
    
    Response:
        {
            "timestamp": "string (ISO format)",
            "production_priorities": [
                {
                    "rank": int,
                    "product_name": "string",
                    "recommended_quantity": float,
                    "demand_forecast": float,
                    "production_cost": float,
                    "priority_score": float
                }
            ],
            "focus_products": [
                {
                    "rank": int,
                    "product_name": "string",
                    "recommended_quantity": float,
                    "focus_rationale": "string"
                }
            ],
            "resource_allocation": {
                "by_product": {
                    "product_name": {
                        "quantity": float,
                        "percentage": float,
                        "estimated_cost": float,
                        "estimated_revenue": float
                    }
                },
                "total_quantity": float,
                "total_cost": float,
                "total_revenue": float
            },
            "optimization_metrics": {
                "total_revenue": float,
                "total_cost": float,
                "profit": float,
                "roi": float
            },
            "constraints_applied": bool,
            "optimization_success": bool,
            "processing_time_seconds": float,
            "num_products": int
        }
    
    Status Codes:
        200: Success
        400: Bad Request
        503: Service Not Ready
        500: Internal Server Error
    """
    try:
        # Get request data
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.warning(f"JSON parsing error: {json_error}")
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid JSON or missing Content-Type header',
                'status_code': 400
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON',
                'status_code': 400
            }), 400
        
        # Validate request
        try:
            validated_data = business_optimizer_request_schema.load(data)
        except ValidationError as err:
            logger.warning(f"Validation error: {err.messages}")
            return jsonify({
                'error': 'Validation Error',
                'message': err.messages,
                'status_code': 400
            }), 400
        
        # Extract parameters
        product_portfolio = validated_data['product_portfolio']
        rl_strategy_outputs = validated_data.get('rl_strategy_outputs')
        lstm_forecast_outputs = validated_data.get('lstm_forecast_outputs')
        constraints = validated_data.get('constraints')
        revenue_weight = validated_data.get('revenue_weight', 0.7)
        cost_weight = validated_data.get('cost_weight', 0.3)
        
        logger.info(f"Processing business_optimizer request (products: {len(product_portfolio)})")
        
        # Initialize service
        service = BusinessManagerService()
        
        # Optimize business (will use fallback if model not loaded)
        try:
            result = service.optimize_business(
                product_portfolio=product_portfolio,
                rl_strategy_outputs=rl_strategy_outputs,
                lstm_forecast_outputs=lstm_forecast_outputs,
                constraints=constraints,
                revenue_weight=revenue_weight,
                cost_weight=cost_weight
            )
            
            # Validate response
            validated_response = business_optimizer_response_schema.dump(result)
            
            return jsonify(validated_response), 200
            
        except TimeoutError:
            return jsonify({
                'error': 'Timeout',
                'message': 'Business optimization exceeded time limit. Please try with fewer products.',
                'status_code': 504
            }), 504
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({
            'error': 'Invalid Input',
            'message': str(e),
            'status_code': 400
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return jsonify({
            'error': 'Service Error',
            'message': str(e),
            'status_code': 500
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in optimize_business: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500


@api_bp.route('/business_optimizer/scenarios', methods=['POST'])
def analyze_business_scenarios():
    """
    Analyze multiple what-if scenarios for business optimization.
    
    Request Body:
        {
            "product_portfolio": [...] (required),
            "scenarios": [
                {
                    "name": "string (optional)",
                    "constraints": {...} (optional),
                    "rl_outputs": {...} (optional),
                    "lstm_outputs": {...} (optional)
                }
            ] (required, max 10 scenarios)
        }
    
    Response:
        {
            "scenarios": [
                {
                    "name": "string",
                    "production_priorities": [...],
                    "focus_products": [...],
                    "metrics": {...},
                    "success": bool
                }
            ],
            "comparison": {
                "best_scenario": "string",
                "best_scenario_profit": float,
                "comparison_metrics": {...},
                "num_successful_scenarios": int
            },
            "num_scenarios": int
        }
    
    Status Codes:
        200: Success
        400: Bad Request
        503: Service Not Ready
        500: Internal Server Error
    """
    try:
        # Get request data
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.warning(f"JSON parsing error: {json_error}")
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid JSON or missing Content-Type header',
                'status_code': 400
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON',
                'status_code': 400
            }), 400
        
        # Validate required fields
        if 'product_portfolio' not in data or 'scenarios' not in data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'product_portfolio and scenarios fields are required',
                'status_code': 400
            }), 400
        
        product_portfolio = data['product_portfolio']
        scenarios = data['scenarios']
        
        if not isinstance(scenarios, list) or len(scenarios) == 0:
            return jsonify({
                'error': 'Bad Request',
                'message': 'scenarios must be a non-empty list',
                'status_code': 400
            }), 400
        
        if len(scenarios) > 10:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Maximum 10 scenarios allowed',
                'status_code': 400
            }), 400
        
        logger.info(f"Processing business scenario analysis (scenarios: {len(scenarios)})")
        
        # Initialize service
        service = BusinessManagerService()
        
        # Check if model is loaded
        if not service.model_loaded:
            return jsonify({
                'error': 'Service Not Ready',
                'message': 'Business optimizer model not loaded. Please train or load a model first.',
                'status_code': 503
            }), 503
        
        # Analyze scenarios
        result = service.analyze_scenarios(
            product_portfolio=product_portfolio,
            scenarios=scenarios
        )
        
        return jsonify(result), 200
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({
            'error': 'Invalid Input',
            'message': str(e),
            'status_code': 400
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_business_scenarios: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500



@api_bp.route('/explain', methods=['POST'])
def explain_model():
    """
    Explain model predictions using SHAP.
    
    Request Body:
        {
            "model_type": "rl | lstm | regression" (required),
            "instance": [float] or [[float]] (required, input instance to explain),
            "top_n": int (optional, default: 10, number of top features),
            "include_visualizations": bool (optional, default: false),
            "background_data": [[float]] (optional, background dataset for SHAP),
            "explanation_type": "local | global" (optional, default: "local")
        }
    
    Response (local explanation):
        {
            "model_type": "string",
            "explanation_type": "local",
            "base_value": float,
            "top_features": [
                {
                    "feature": "string",
                    "shap_value": float,
                    "contribution_pct": float
                }
            ],
            "total_features": int,
            "visualizations": {
                "force_plot": "string (base64)",
                "waterfall_chart": "string (base64)",
                "bar_chart": "string (base64)"
            } (optional),
            "timestamp": "string (ISO format)"
        }
    
    Response (global explanation):
        {
            "model_type": "string",
            "explanation_type": "global",
            "top_features": [
                {
                    "feature": "string",
                    "importance": float,
                    "importance_pct": float
                }
            ],
            "total_features": int,
            "num_samples_analyzed": int,
            "timestamp": "string (ISO format)"
        }
    
    Status Codes:
        200: Success
        400: Bad Request
        503: Service Not Ready
        500: Internal Server Error
    """
    try:
        # Get request data
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.warning(f"JSON parsing error: {json_error}")
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid JSON or missing Content-Type header',
                'status_code': 400
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON',
                'status_code': 400
            }), 400
        
        # Validate required fields
        if 'model_type' not in data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'model_type field is required',
                'status_code': 400
            }), 400
        
        if 'instance' not in data and data.get('explanation_type', 'local') == 'local':
            return jsonify({
                'error': 'Bad Request',
                'message': 'instance field is required for local explanations',
                'status_code': 400
            }), 400
        
        # Extract parameters
        model_type = data['model_type'].lower()
        
        if model_type not in ['rl', 'lstm', 'regression']:
            return jsonify({
                'error': 'Bad Request',
                'message': 'model_type must be one of: rl, lstm, regression',
                'status_code': 400
            }), 400
        
        explanation_type = data.get('explanation_type', 'local').lower()
        
        if explanation_type not in ['local', 'global']:
            return jsonify({
                'error': 'Bad Request',
                'message': 'explanation_type must be one of: local, global',
                'status_code': 400
            }), 400
        
        top_n = data.get('top_n', 10)
        include_visualizations = data.get('include_visualizations', False)
        
        # Convert data to numpy arrays
        try:
            if explanation_type == 'local':
                instance = np.array(data['instance'], dtype=np.float32)
            else:
                instance = None
            
            background_data = None
            if 'background_data' in data and data['background_data']:
                background_data = np.array(data['background_data'], dtype=np.float32)
        
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'Bad Request',
                'message': f'Invalid data format: {str(e)}',
                'status_code': 400
            }), 400
        
        logger.info(f"Processing explain request (model_type: {model_type}, explanation_type: {explanation_type})")
        
        # Initialize model transparency service
        from src.services.model_transparency_service import ModelTransparencyService
        
        # Load models based on model_type
        rl_agent = None
        lstm_forecaster = None
        regression_optimizer = None
        
        try:
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
        
        except FileNotFoundError as e:
            return jsonify({
                'error': 'Service Not Ready',
                'message': f'Model not found: {str(e)}. Please train or load the model first.',
                'status_code': 503
            }), 503
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return jsonify({
                'error': 'Service Error',
                'message': f'Failed to load model: {str(e)}',
                'status_code': 500
            }), 500
        
        # Create transparency service
        transparency_service = ModelTransparencyService(
            rl_agent=rl_agent,
            lstm_forecaster=lstm_forecaster,
            regression_optimizer=regression_optimizer
        )
        
        # Generate explanation
        try:
            if explanation_type == 'local':
                result = transparency_service.explain_prediction(
                    model_type=model_type,
                    instance=instance,
                    top_n=top_n,
                    include_visualizations=include_visualizations,
                    background_data=background_data
                )
                result['explanation_type'] = 'local'
            else:  # global
                if background_data is None:
                    return jsonify({
                        'error': 'Bad Request',
                        'message': 'background_data is required for global explanations',
                        'status_code': 400
                    }), 400
                
                result = transparency_service.explain_global(
                    model_type=model_type,
                    data=background_data,
                    top_n=top_n
                )
                result['explanation_type'] = 'global'
            
            return jsonify(result), 200
            
        except TimeoutError:
            return jsonify({
                'error': 'Timeout',
                'message': 'Explanation generation exceeded time limit. Please try with a smaller dataset.',
                'status_code': 504
            }), 504
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return jsonify({
            'error': 'Invalid Input',
            'message': str(e),
            'status_code': 400
        }), 400
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return jsonify({
            'error': 'Service Error',
            'message': str(e),
            'status_code': 500
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in explain_model: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500


@api_bp.route('/explain/batch', methods=['POST'])
def explain_batch():
    """
    Explain multiple predictions in batch.
    
    Request Body:
        {
            "model_type": "rl | lstm | regression" (required),
            "instances": [[float]] or [[[float]]] (required, list of instances),
            "top_n": int (optional, default: 10),
            "background_data": [[float]] (optional)
        }
    
    Response:
        {
            "model_type": "string",
            "explanations": [
                {
                    "base_value": float,
                    "top_features": [...],
                    "total_features": int,
                    "timestamp": "string"
                }
            ],
            "num_instances": int,
            "processing_time_seconds": float
        }
    
    Status Codes:
        200: Success
        400: Bad Request
        503: Service Not Ready
        500: Internal Server Error
    """
    try:
        # Get request data
        try:
            data = request.get_json()
        except Exception as json_error:
            logger.warning(f"JSON parsing error: {json_error}")
            return jsonify({
                'error': 'Bad Request',
                'message': 'Invalid JSON or missing Content-Type header',
                'status_code': 400
            }), 400
        
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body must be JSON',
                'status_code': 400
            }), 400
        
        # Validate required fields
        if 'model_type' not in data or 'instances' not in data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'model_type and instances fields are required',
                'status_code': 400
            }), 400
        
        model_type = data['model_type'].lower()
        
        if model_type not in ['rl', 'lstm', 'regression']:
            return jsonify({
                'error': 'Bad Request',
                'message': 'model_type must be one of: rl, lstm, regression',
                'status_code': 400
            }), 400
        
        top_n = data.get('top_n', 10)
        
        # Convert data to numpy arrays
        try:
            instances = [np.array(inst, dtype=np.float32) for inst in data['instances']]
            
            background_data = None
            if 'background_data' in data and data['background_data']:
                background_data = np.array(data['background_data'], dtype=np.float32)
        
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'Bad Request',
                'message': f'Invalid data format: {str(e)}',
                'status_code': 400
            }), 400
        
        if len(instances) > 100:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Maximum 100 instances allowed per batch request',
                'status_code': 400
            }), 400
        
        logger.info(f"Processing batch explain request (model_type: {model_type}, instances: {len(instances)})")
        
        # Initialize model transparency service
        from src.services.model_transparency_service import ModelTransparencyService
        
        # Load models
        rl_agent = None
        lstm_forecaster = None
        regression_optimizer = None
        
        try:
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
        
        except FileNotFoundError as e:
            return jsonify({
                'error': 'Service Not Ready',
                'message': f'Model not found: {str(e)}. Please train or load the model first.',
                'status_code': 503
            }), 503
        
        # Create transparency service
        transparency_service = ModelTransparencyService(
            rl_agent=rl_agent,
            lstm_forecaster=lstm_forecaster,
            regression_optimizer=regression_optimizer
        )
        
        # Generate batch explanations
        start_time = time.time()
        
        explanations = transparency_service.batch_explain(
            model_type=model_type,
            instances=instances,
            top_n=top_n,
            background_data=background_data
        )
        
        processing_time = time.time() - start_time
        
        result = {
            'model_type': model_type,
            'explanations': explanations,
            'num_instances': len(instances),
            'processing_time_seconds': processing_time
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in explain_batch: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500
