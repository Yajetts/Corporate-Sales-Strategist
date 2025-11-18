"""Middleware for request validation and processing"""

import logging
import time
from flask import request, jsonify, g
from functools import wraps
from typing import Optional

logger = logging.getLogger(__name__)


def request_logger(f):
    """
    Middleware to log incoming requests.
    
    Usage:
        @api_bp.route('/endpoint')
        @request_logger
        def endpoint():
            return jsonify({'message': 'Success'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Log request details
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
        
        # Store start time
        g.start_time = time.time()
        
        # Execute endpoint
        response = f(*args, **kwargs)
        
        # Log response time
        elapsed_time = time.time() - g.start_time
        logger.info(f"Response: {request.method} {request.path} completed in {elapsed_time:.3f}s")
        
        return response
    
    return decorated_function


def validate_content_type(required_type='application/json'):
    """
    Middleware to validate request Content-Type header.
    
    Args:
        required_type: Required Content-Type value
    
    Usage:
        @api_bp.route('/endpoint', methods=['POST'])
        @validate_content_type('application/json')
        def endpoint():
            return jsonify({'message': 'Success'})
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method in ['POST', 'PUT', 'PATCH']:
                content_type = request.headers.get('Content-Type', '')
                
                # Check if content type matches (allow charset specification)
                if not content_type.startswith(required_type):
                    return jsonify({
                        'error': 'Unsupported Media Type',
                        'message': f'Content-Type must be {required_type}',
                        'status_code': 415
                    }), 415
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def rate_limit(max_requests: int, window_seconds: int = 60):
    """
    Simple rate limiting middleware.
    
    Args:
        max_requests: Maximum number of requests allowed
        window_seconds: Time window in seconds
    
    Usage:
        @api_bp.route('/endpoint')
        @rate_limit(max_requests=100, window_seconds=60)
        def endpoint():
            return jsonify({'message': 'Success'})
    """
    # Store request timestamps per IP
    request_history = {}
    
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            # Initialize history for this IP if not exists
            if client_ip not in request_history:
                request_history[client_ip] = []
            
            # Remove old requests outside the window
            request_history[client_ip] = [
                timestamp for timestamp in request_history[client_ip]
                if current_time - timestamp < window_seconds
            ]
            
            # Check if limit exceeded
            if len(request_history[client_ip]) >= max_requests:
                return jsonify({
                    'error': 'Rate Limit Exceeded',
                    'message': f'Maximum {max_requests} requests per {window_seconds} seconds allowed',
                    'status_code': 429,
                    'retry_after': window_seconds
                }), 429
            
            # Add current request
            request_history[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def add_response_headers(headers: dict):
    """
    Middleware to add custom headers to response.
    
    Args:
        headers: Dictionary of headers to add
    
    Usage:
        @api_bp.route('/endpoint')
        @add_response_headers({'X-Custom-Header': 'value'})
        def endpoint():
            return jsonify({'message': 'Success'})
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = f(*args, **kwargs)
            
            # Handle different response types
            if isinstance(response, tuple):
                resp_obj, status_code = response[0], response[1]
                for key, value in headers.items():
                    resp_obj.headers[key] = value
                return resp_obj, status_code
            else:
                for key, value in headers.items():
                    response.headers[key] = value
                return response
        
        return decorated_function
    
    return decorator


def request_id_middleware(app):
    """
    Add unique request ID to each request for tracing.
    
    Args:
        app: Flask application instance
    """
    import uuid
    
    @app.before_request
    def add_request_id():
        g.request_id = str(uuid.uuid4())
        logger.info(f"Request ID: {g.request_id}")
    
    @app.after_request
    def add_request_id_header(response):
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        return response


def error_handler_middleware(app):
    """
    Global error handlers for the application.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': str(error),
            'status_code': 400
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Authentication required',
            'status_code': 401
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({
            'error': 'Forbidden',
            'message': 'Access denied',
            'status_code': 403
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource was not found',
            'status_code': 404
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'error': 'Method Not Allowed',
            'message': 'The method is not allowed for the requested URL',
            'status_code': 405
        }), 405
    
    @app.errorhandler(415)
    def unsupported_media_type(error):
        return jsonify({
            'error': 'Unsupported Media Type',
            'message': 'The media type is not supported',
            'status_code': 415
        }), 415
    
    @app.errorhandler(429)
    def too_many_requests(error):
        return jsonify({
            'error': 'Too Many Requests',
            'message': 'Rate limit exceeded',
            'status_code': 429
        }), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error(f"Internal server error: {error}", exc_info=True)
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500
    
    @app.errorhandler(503)
    def service_unavailable(error):
        return jsonify({
            'error': 'Service Unavailable',
            'message': 'The service is temporarily unavailable',
            'status_code': 503
        }), 503
