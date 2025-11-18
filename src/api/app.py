"""Main Flask application for Sales Strategist API"""

import logging
from flask import Flask, jsonify
from flask_cors import CORS
from src.utils.config import Config, get_config
from src.api.routes import api_bp
from src.api.async_routes import async_bp
from src.dashboard.routes import dashboard_bp
from src.api.health import health_bp
from src.api.middleware import request_id_middleware, error_handler_middleware
from src.api.swagger import register_swagger_routes
from src.api.database import initialize_databases, cleanup_databases, postgres_manager, mongodb_manager
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name=None):
    """
    Application factory for creating Flask app.
    
    Args:
        config_name: Configuration environment name
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Ensure necessary directories exist
    config.ensure_directories()
    
    # Enable CORS with configuration
    cors_config = {
        'origins': config.CORS_ORIGINS if hasattr(config, 'CORS_ORIGINS') else '*',
        'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        'allow_headers': ['Content-Type', 'Authorization', 'X-API-Key', 'X-Request-ID'],
        'expose_headers': ['X-Request-ID'],
        'supports_credentials': True
    }
    CORS(app, **cors_config)
    
    # Register middleware
    request_id_middleware(app)
    error_handler_middleware(app)
    
    # Register blueprints
    api_prefix = config.API_PREFIX
    app.register_blueprint(api_bp, url_prefix=api_prefix)
    app.register_blueprint(async_bp, url_prefix=f'{api_prefix}/async')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(health_bp, url_prefix=api_prefix)
    
    # Register Swagger/OpenAPI documentation
    register_swagger_routes(app, api_prefix)
    
    # Initialize databases
    try:
        initialize_databases()
        logger.info("Databases initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}. Some features may not work.")
    
    # Register cleanup on app shutdown
    atexit.register(cleanup_databases)
    
    # Root endpoint - redirect to dashboard
    @app.route('/')
    def index():
        from flask import redirect, url_for
        return redirect(url_for('dashboard.index'))
    
    # Database health check endpoint
    @app.route(f'{api_prefix}/db/health')
    def db_health():
        postgres_healthy = postgres_manager.health_check()
        mongodb_healthy = mongodb_manager.health_check()
        
        overall_healthy = postgres_healthy and mongodb_healthy
        
        return jsonify({
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'databases': {
                'postgresql': {
                    'status': 'healthy' if postgres_healthy else 'unhealthy'
                },
                'mongodb': {
                    'status': 'healthy' if mongodb_healthy else 'unhealthy'
                }
            }
        }), 200 if overall_healthy else 503
    
    # Error handlers
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
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500
    
    logger.info(f"Flask app created with config: {config_name or 'development'}")
    logger.info(f"API prefix: {api_prefix}")
    
    return app


# Create app instance
app = create_app()


if __name__ == '__main__':
    config = Config()
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG
    )
