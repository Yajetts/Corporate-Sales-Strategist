"""Authentication and authorization middleware for API"""

import os
import jwt
import logging
from functools import wraps
from flask import request, jsonify
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class AuthManager:
    """Manages API authentication and authorization"""
    
    def __init__(self, secret_key: Optional[str] = None, api_keys: Optional[set] = None):
        """
        Initialize authentication manager.
        
        Args:
            secret_key: Secret key for JWT signing
            api_keys: Set of valid API keys
        """
        self.secret_key = secret_key or os.getenv('SECRET_KEY', 'dev-secret-key')
        self.api_keys = api_keys or self._load_api_keys()
        self.jwt_algorithm = 'HS256'
        self.jwt_expiration_hours = 24
    
    def _load_api_keys(self) -> set:
        """Load API keys from environment"""
        api_keys_str = os.getenv('API_KEYS', '')
        if api_keys_str:
            return set(api_keys_str.split(','))
        return set()
    
    def generate_jwt_token(self, user_id: str, additional_claims: Optional[Dict] = None) -> str:
        """
        Generate JWT token for user.
        
        Args:
            user_id: User identifier
            additional_claims: Additional claims to include in token
            
        Returns:
            JWT token string
        """
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours),
            'iat': datetime.utcnow()
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.jwt_algorithm)
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def verify_api_key(self, api_key: str) -> bool:
        """
        Verify API key.
        
        Args:
            api_key: API key to verify
            
        Returns:
            True if valid, False otherwise
        """
        return api_key in self.api_keys
    
    def authenticate_request(self) -> tuple[bool, Optional[Dict], Optional[str]]:
        """
        Authenticate incoming request using JWT or API key.
        
        Returns:
            Tuple of (is_authenticated, user_data, error_message)
        """
        # Check for API key in header
        api_key = request.headers.get('X-API-Key')
        if api_key:
            if self.verify_api_key(api_key):
                return True, {'auth_method': 'api_key'}, None
            else:
                return False, None, 'Invalid API key'
        
        # Check for JWT token in Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
                payload = self.verify_jwt_token(token)
                if payload:
                    return True, payload, None
                else:
                    return False, None, 'Invalid or expired token'
            else:
                return False, None, 'Invalid Authorization header format'
        
        # No authentication provided
        return False, None, 'No authentication credentials provided'


# Global auth manager instance
auth_manager = AuthManager()


def require_auth(f):
    """
    Decorator to require authentication for endpoint.
    
    Usage:
        @api_bp.route('/protected')
        @require_auth
        def protected_endpoint():
            return jsonify({'message': 'Success'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        is_authenticated, user_data, error_message = auth_manager.authenticate_request()
        
        if not is_authenticated:
            return jsonify({
                'error': 'Unauthorized',
                'message': error_message,
                'status_code': 401
            }), 401
        
        # Add user data to request context
        request.user_data = user_data
        
        return f(*args, **kwargs)
    
    return decorated_function


def optional_auth(f):
    """
    Decorator for optional authentication.
    Adds user data to request if authenticated, but doesn't require it.
    
    Usage:
        @api_bp.route('/public')
        @optional_auth
        def public_endpoint():
            if hasattr(request, 'user_data'):
                # User is authenticated
                pass
            return jsonify({'message': 'Success'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        is_authenticated, user_data, _ = auth_manager.authenticate_request()
        
        if is_authenticated:
            request.user_data = user_data
        
        return f(*args, **kwargs)
    
    return decorated_function
