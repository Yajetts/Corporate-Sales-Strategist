"""Main Flask application for Sales Strategist API"""

import logging
import os
import uuid
from flask import Flask, jsonify
from flask_cors import CORS

try:
    from dotenv import load_dotenv
except Exception:  # optional at runtime; declared in requirements
    load_dotenv = None
from src.utils.config import Config, get_config
from src.api.post_analysis_routes import post_analysis_bp
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
    # Load local env files early so config + integrations (LLM, DB, etc.) work in dev.
    if load_dotenv is not None:
        load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=False)
        load_dotenv(dotenv_path=os.getenv("FLASK_ENV_FILE", ".flaskenv"), override=False)

    app = Flask(__name__)

    # Unique per-process identifier used to tag cached module outputs.
    # This lets the dashboard reliably detect "modules ran in this session"
    # without being fooled by stale JSON artifacts on disk.
    app.config["BOOT_ID"] = uuid.uuid4().hex
    
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

    enable_core_routes = os.getenv('API_ENABLE_CORE_ROUTES', '1').strip().lower() not in {'0', 'false', 'no'}
    enable_async_routes = os.getenv('API_ENABLE_ASYNC_ROUTES', '1').strip().lower() not in {'0', 'false', 'no'}
    enable_swagger = os.getenv('API_ENABLE_SWAGGER', '1').strip().lower() not in {'0', 'false', 'no'}

    # Core routes import ML/torch-heavy modules via services; keep them optional so
    # Docker can run a slim post-analysis + dashboard stack.
    if enable_core_routes:
        from src.api.routes import api_bp
        app.register_blueprint(api_bp, url_prefix=api_prefix)

    if enable_async_routes:
        from src.api.async_routes import async_bp
        app.register_blueprint(async_bp, url_prefix=f'{api_prefix}/async')

    app.register_blueprint(post_analysis_bp, url_prefix=api_prefix)
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    app.register_blueprint(health_bp, url_prefix=api_prefix)
    
    # Register Swagger/OpenAPI documentation
    if enable_swagger:
        register_swagger_routes(app, api_prefix)
    
    # Initialize databases
    try:
        ok = initialize_databases()
        if ok:
            logger.info("Databases initialized successfully")
        else:
            logger.warning("Databases unavailable; running in degraded mode")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}. Some features may not work.")
    
    # Register cleanup on app shutdown
    atexit.register(cleanup_databases)
    
    # Root endpoint
    @app.route('/')
    def index():
        # Keep this endpoint lightweight and DB-independent so tests and
        # health checks work even when optional services (e.g., Postgres) are
        # unavailable.
        return jsonify({
            'service': 'ai-sales-strategist-api',
            'version': getattr(config, 'VERSION', '1.0.0'),
            'status': 'running'
        })
    
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
