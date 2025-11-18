"""REST API endpoints and gateway"""

from src.api.app import app, create_app
from src.api.routes import api_bp

__all__ = ['app', 'create_app', 'api_bp']
