"""Async task management routes"""

import logging
from flask import Blueprint, request, jsonify
from celery.result import AsyncResult
from src.api.celery_app import celery
from src.api.tasks import (
    analyze_company_async,
    analyze_market_async,
    generate_strategy_async,
    monitor_performance_async,
    optimize_business_async,
    explain_model_async
)
from marshmallow import ValidationError
from src.api.schemas import (
    AnalyzeCompanyRequestSchema,
    MarketAnalysisRequestSchema,
    StrategyRequestSchema,
    BusinessOptimizerRequestSchema
)

logger = logging.getLogger(__name__)

# Create blueprint for async routes
async_bp = Blueprint('async', __name__)

# Initialize schemas
analyze_company_request_schema = AnalyzeCompanyRequestSchema()
market_analysis_request_schema = MarketAnalysisRequestSchema()
strategy_request_schema = StrategyRequestSchema()
business_optimizer_request_schema = BusinessOptimizerRequestSchema()


@async_bp.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """
    Get status of an async task.
    
    Path Parameters:
        task_id: Unique task identifier
    
    Response:
        {
            "task_id": "string",
            "status": "PENDING | STARTED | SUCCESS | FAILURE | RETRY",
            "result": {...} (if completed),
            "error": "string" (if failed),
            "progress": {
                "current": int,
                "total": int,
                "percent": float
            } (if available)
        }
    
    Status Codes:
        200: Success
        404: Task not found
        500: Internal Server Error
    """
    try:
        task = AsyncResult(task_id, app=celery)
        
        response = {
            'task_id': task_id,
            'status': task.state
        }
        
        if task.state == 'PENDING':
            response['message'] = 'Task is waiting to be executed'
        
        elif task.state == 'STARTED':
            response['message'] = 'Task is currently running'
            # Add progress info if available
            if task.info:
                response['progress'] = task.info
        
        elif task.state == 'SUCCESS':
            response['message'] = 'Task completed successfully'
            response['result'] = task.result
        
        elif task.state == 'FAILURE':
            response['message'] = 'Task failed'
            response['error'] = str(task.info)
        
        elif task.state == 'RETRY':
            response['message'] = 'Task is being retried'
            response['error'] = str(task.info)
        
        else:
            response['message'] = f'Task state: {task.state}'
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error getting task status: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500


@async_bp.route('/tasks/<task_id>/result', methods=['GET'])
def get_task_result(task_id):
    """
    Get result of a completed async task.
    
    Path Parameters:
        task_id: Unique task identifier
    
    Query Parameters:
        wait: bool (optional, default: false) - Wait for task to complete
        timeout: int (optional, default: 30) - Timeout in seconds if waiting
    
    Response:
        Task result (varies by task type)
    
    Status Codes:
        200: Success
        202: Task not yet completed (if not waiting)
        404: Task not found
        408: Timeout waiting for result
        500: Task failed or internal error
    """
    try:
        task = AsyncResult(task_id, app=celery)
        
        # Check if we should wait for result
        wait = request.args.get('wait', 'false').lower() == 'true'
        timeout = int(request.args.get('timeout', 30))
        
        if wait:
            try:
                result = task.get(timeout=timeout)
                return jsonify(result), 200
            except TimeoutError:
                return jsonify({
                    'error': 'Timeout',
                    'message': f'Task did not complete within {timeout} seconds',
                    'status_code': 408,
                    'task_id': task_id,
                    'status': task.state
                }), 408
        else:
            if task.ready():
                if task.successful():
                    return jsonify(task.result), 200
                else:
                    return jsonify({
                        'error': 'Task Failed',
                        'message': str(task.info),
                        'status_code': 500,
                        'task_id': task_id
                    }), 500
            else:
                return jsonify({
                    'message': 'Task not yet completed',
                    'status_code': 202,
                    'task_id': task_id,
                    'status': task.state
                }), 202
        
    except Exception as e:
        logger.error(f"Error getting task result: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500


@async_bp.route('/tasks/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    """
    Cancel a running async task.
    
    Path Parameters:
        task_id: Unique task identifier
    
    Response:
        {
            "task_id": "string",
            "status": "string",
            "message": "string"
        }
    
    Status Codes:
        200: Task cancelled successfully
        400: Task cannot be cancelled
        404: Task not found
        500: Internal Server Error
    """
    try:
        task = AsyncResult(task_id, app=celery)
        
        if task.state in ['PENDING', 'STARTED', 'RETRY']:
            # Revoke the task
            task.revoke(terminate=True, signal='SIGTERM')
            
            return jsonify({
                'task_id': task_id,
                'status': 'CANCELLED',
                'message': 'Task cancellation requested'
            }), 200
        else:
            return jsonify({
                'error': 'Bad Request',
                'message': f'Task cannot be cancelled (current state: {task.state})',
                'status_code': 400,
                'task_id': task_id
            }), 400
        
    except Exception as e:
        logger.error(f"Error cancelling task: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500


@async_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """
    List all active tasks.
    
    Query Parameters:
        limit: int (optional, default: 100) - Maximum number of tasks to return
        status: string (optional) - Filter by task status
    
    Response:
        {
            "tasks": [
                {
                    "task_id": "string",
                    "status": "string",
                    "name": "string"
                }
            ],
            "total": int
        }
    
    Status Codes:
        200: Success
        500: Internal Server Error
    """
    try:
        limit = int(request.args.get('limit', 100))
        status_filter = request.args.get('status')
        
        # Get active tasks from Celery
        inspect = celery.control.inspect()
        
        active_tasks = []
        
        # Get active (running) tasks
        active = inspect.active()
        if active:
            for worker, tasks in active.items():
                for task in tasks[:limit]:
                    if not status_filter or status_filter.upper() == 'STARTED':
                        active_tasks.append({
                            'task_id': task['id'],
                            'name': task['name'],
                            'status': 'STARTED',
                            'worker': worker
                        })
        
        # Get scheduled tasks
        scheduled = inspect.scheduled()
        if scheduled:
            for worker, tasks in scheduled.items():
                for task in tasks[:limit]:
                    if not status_filter or status_filter.upper() == 'PENDING':
                        active_tasks.append({
                            'task_id': task['request']['id'],
                            'name': task['request']['name'],
                            'status': 'PENDING',
                            'worker': worker
                        })
        
        # Get reserved tasks
        reserved = inspect.reserved()
        if reserved:
            for worker, tasks in reserved.items():
                for task in tasks[:limit]:
                    if not status_filter or status_filter.upper() == 'PENDING':
                        active_tasks.append({
                            'task_id': task['id'],
                            'name': task['name'],
                            'status': 'PENDING',
                            'worker': worker
                        })
        
        return jsonify({
            'tasks': active_tasks[:limit],
            'total': len(active_tasks)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500


# Async endpoint wrappers for each AI operation

@async_bp.route('/analyze_company/async', methods=['POST'])
def analyze_company_async_endpoint():
    """
    Submit company analysis task for async processing.
    
    Request Body: Same as /analyze_company
    
    Response:
        {
            "task_id": "string",
            "status": "PENDING",
            "message": "Task submitted successfully",
            "status_url": "/api/v1/async/tasks/{task_id}",
            "result_url": "/api/v1/async/tasks/{task_id}/result"
        }
    
    Status Codes:
        202: Task accepted
        400: Bad Request
        500: Internal Server Error
    """
    try:
        data = request.get_json()
        
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
            return jsonify({
                'error': 'Validation Error',
                'message': err.messages,
                'status_code': 400
            }), 400
        
        # Submit task
        task = analyze_company_async.delay(
            text=validated_data['text'],
            source_type=validated_data.get('source_type')
        )
        
        return jsonify({
            'task_id': task.id,
            'status': 'PENDING',
            'message': 'Task submitted successfully',
            'status_url': f'/api/v1/async/tasks/{task.id}',
            'result_url': f'/api/v1/async/tasks/{task.id}/result'
        }), 202
        
    except Exception as e:
        logger.error(f"Error submitting async task: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500


@async_bp.route('/market_analysis/async', methods=['POST'])
def analyze_market_async_endpoint():
    """Submit market analysis task for async processing."""
    try:
        data = request.get_json()
        
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
            return jsonify({
                'error': 'Validation Error',
                'message': err.messages,
                'status_code': 400
            }), 400
        
        # Submit task
        task = analyze_market_async.delay(
            market_data=validated_data['market_data'],
            entity_ids=validated_data.get('entity_ids'),
            auto_select_clusters=validated_data.get('auto_select_clusters', True),
            similarity_threshold=validated_data.get('similarity_threshold', 0.7),
            top_k_links=validated_data.get('top_k_links', 20)
        )
        
        return jsonify({
            'task_id': task.id,
            'status': 'PENDING',
            'message': 'Task submitted successfully',
            'status_url': f'/api/v1/async/tasks/{task.id}',
            'result_url': f'/api/v1/async/tasks/{task.id}/result'
        }), 202
        
    except Exception as e:
        logger.error(f"Error submitting async task: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500


@async_bp.route('/strategy/async', methods=['POST'])
def generate_strategy_async_endpoint():
    """Submit strategy generation task for async processing."""
    try:
        data = request.get_json()
        
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
            return jsonify({
                'error': 'Validation Error',
                'message': err.messages,
                'status_code': 400
            }), 400
        
        # Submit task
        task = generate_strategy_async.delay(
            market_state=validated_data['market_state'],
            context=validated_data.get('context'),
            include_explanation=validated_data.get('include_explanation', True),
            deterministic=validated_data.get('deterministic', True)
        )
        
        return jsonify({
            'task_id': task.id,
            'status': 'PENDING',
            'message': 'Task submitted successfully',
            'status_url': f'/api/v1/async/tasks/{task.id}',
            'result_url': f'/api/v1/async/tasks/{task.id}/result'
        }), 202
        
    except Exception as e:
        logger.error(f"Error submitting async task: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500


@async_bp.route('/business_optimizer/async', methods=['POST'])
def optimize_business_async_endpoint():
    """Submit business optimization task for async processing."""
    try:
        data = request.get_json()
        
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
            return jsonify({
                'error': 'Validation Error',
                'message': err.messages,
                'status_code': 400
            }), 400
        
        # Submit task
        task = optimize_business_async.delay(
            product_portfolio=validated_data['product_portfolio'],
            rl_strategy_outputs=validated_data.get('rl_strategy_outputs'),
            lstm_forecast_outputs=validated_data.get('lstm_forecast_outputs'),
            constraints=validated_data.get('constraints'),
            revenue_weight=validated_data.get('revenue_weight', 0.7),
            cost_weight=validated_data.get('cost_weight', 0.3)
        )
        
        return jsonify({
            'task_id': task.id,
            'status': 'PENDING',
            'message': 'Task submitted successfully',
            'status_url': f'/api/v1/async/tasks/{task.id}',
            'result_url': f'/api/v1/async/tasks/{task.id}/result'
        }), 202
        
    except Exception as e:
        logger.error(f"Error submitting async task: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e),
            'status_code': 500
        }), 500
