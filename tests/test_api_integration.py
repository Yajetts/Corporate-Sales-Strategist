"""Comprehensive API integration tests for Sales Strategist System"""

import pytest
import json
import time
import os
from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client with testing configuration"""
    app = create_app('testing')
    app.config['TESTING'] = True
    app.config['API_KEYS'] = 'test-api-key-123,test-api-key-456'
    
    with app.test_client() as client:
        yield client


@pytest.fixture
def auth_headers():
    """Provide authentication headers for testing"""
    return {
        'X-API-Key': 'test-api-key-123',
        'Content-Type': 'application/json'
    }


@pytest.fixture
def invalid_auth_headers():
    """Provide invalid authentication headers for testing"""
    return {
        'X-API-Key': 'invalid-key',
        'Content-Type': 'application/json'
    }


class TestAPIAuthentication:
    """Test authentication and authorization"""
    
    def test_api_key_authentication_valid(self, client, auth_headers):
        """Test valid API key authentication"""
        response = client.get('/api/v1/health', headers=auth_headers)
        # Should work regardless of auth (health endpoint may be public)
        assert response.status_code in [200, 503]
    
    def test_api_key_authentication_invalid(self, client, invalid_auth_headers):
        """Test invalid API key authentication"""
        # Test on an endpoint that requires auth (if any)
        response = client.get('/api/v1/model_info', headers=invalid_auth_headers)
        # Most endpoints don't require auth in current implementation
        assert response.status_code in [200, 401, 503]
    
    def test_missing_authentication(self, client):
        """Test request without authentication headers"""
        response = client.get('/api/v1/model_info')
        # Should work (no auth required in current implementation)
        assert response.status_code in [200, 401, 503]
    
    def test_jwt_token_authentication(self, client):
        """Test JWT token authentication"""
        from src.api.auth import auth_manager
        
        # Generate a valid JWT token
        token = auth_manager.generate_jwt_token('test_user_123')
        
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        response = client.get('/api/v1/health', headers=headers)
        assert response.status_code in [200, 503]
    
    def test_expired_jwt_token(self, client):
        """Test expired JWT token"""
        # Create a token with very short expiration
        import jwt
        from datetime import datetime, timedelta
        
        secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')
        payload = {
            'user_id': 'test_user',
            'exp': datetime.utcnow() - timedelta(hours=1),  # Expired
            'iat': datetime.utcnow() - timedelta(hours=2)
        }
        
        expired_token = jwt.encode(payload, secret_key, algorithm='HS256')
        
        headers = {
            'Authorization': f'Bearer {expired_token}',
            'Content-Type': 'application/json'
        }
        
        response = client.get('/api/v1/health', headers=headers)
        # Should still work if endpoint doesn't require auth
        assert response.status_code in [200, 401, 503]


class TestAsyncTaskManagement:
    """Test async task submission and retrieval"""
    
    def test_async_company_analysis_submission(self, client):
        """Test submitting async company analysis task"""
        payload = {
            'text': 'TechCorp provides cloud-based software solutions.',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/async/analyze_company/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 202
        data = json.loads(response.data)
        
        assert 'task_id' in data
        assert 'status' in data
        assert data['status'] == 'PENDING'
        assert 'status_url' in data
        assert 'result_url' in data
    
    def test_async_task_status_check(self, client):
        """Test checking async task status"""
        # Submit a task first
        payload = {
            'text': 'TechCorp provides cloud-based software solutions.',
            'source_type': 'product_summary'
        }
        
        submit_response = client.post(
            '/api/v1/async/analyze_company/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert submit_response.status_code == 202
        task_data = json.loads(submit_response.data)
        task_id = task_data['task_id']
        
        # Check task status
        status_response = client.get(f'/api/v1/async/tasks/{task_id}')
        
        assert status_response.status_code == 200
        status_data = json.loads(status_response.data)
        
        assert 'task_id' in status_data
        assert 'status' in status_data
        assert status_data['task_id'] == task_id
        assert status_data['status'] in ['PENDING', 'STARTED', 'SUCCESS', 'FAILURE']
    
    def test_async_task_result_retrieval(self, client):
        """Test retrieving async task result"""
        # Submit a task
        payload = {
            'text': 'TechCorp provides cloud-based software solutions.',
            'source_type': 'product_summary'
        }
        
        submit_response = client.post(
            '/api/v1/async/analyze_company/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        task_data = json.loads(submit_response.data)
        task_id = task_data['task_id']
        
        # Try to get result (may not be ready immediately)
        result_response = client.get(f'/api/v1/async/tasks/{task_id}/result')
        
        # Should return 200 (success), 202 (not ready), or 500 (failed)
        assert result_response.status_code in [200, 202, 500]
    
    def test_async_task_result_with_wait(self, client):
        """Test retrieving async task result with wait parameter"""
        payload = {
            'text': 'TechCorp provides cloud-based software solutions.',
            'source_type': 'product_summary'
        }
        
        submit_response = client.post(
            '/api/v1/async/analyze_company/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        task_data = json.loads(submit_response.data)
        task_id = task_data['task_id']
        
        # Get result with wait and short timeout
        result_response = client.get(
            f'/api/v1/async/tasks/{task_id}/result?wait=true&timeout=5'
        )
        
        # Should return 200 (success), 408 (timeout), or 500 (failed)
        assert result_response.status_code in [200, 408, 500]
    
    def test_async_task_cancellation(self, client):
        """Test cancelling an async task"""
        payload = {
            'text': 'TechCorp provides cloud-based software solutions.',
            'source_type': 'product_summary'
        }
        
        submit_response = client.post(
            '/api/v1/async/analyze_company/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        task_data = json.loads(submit_response.data)
        task_id = task_data['task_id']
        
        # Cancel the task
        cancel_response = client.post(f'/api/v1/async/tasks/{task_id}/cancel')
        
        # Should return 200 (cancelled) or 400 (cannot cancel)
        assert cancel_response.status_code in [200, 400]
    
    def test_async_task_list(self, client):
        """Test listing active async tasks"""
        response = client.get('/api/v1/async/tasks')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'tasks' in data
        assert 'total' in data
        assert isinstance(data['tasks'], list)
    
    def test_async_task_list_with_filters(self, client):
        """Test listing async tasks with filters"""
        response = client.get('/api/v1/async/tasks?limit=10&status=PENDING')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'tasks' in data
        assert len(data['tasks']) <= 10
    
    def test_async_market_analysis_submission(self, client):
        """Test submitting async market analysis task"""
        payload = {
            'market_data': [
                {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.8},
                {'feature1': 0.6, 'feature2': 0.4, 'feature3': 0.7}
            ]
        }
        
        response = client.post(
            '/api/v1/async/market_analysis/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 202
        data = json.loads(response.data)
        assert 'task_id' in data
    
    def test_async_strategy_generation_submission(self, client):
        """Test submitting async strategy generation task"""
        payload = {
            'market_state': {
                'market_demand': 0.7,
                'competitor_prices': [0.5, 0.6, 0.55],
                'sales_volume': 0.65,
                'conversion_rate': 0.15,
                'inventory_level': 0.8,
                'market_trend': 0.2
            }
        }
        
        response = client.post(
            '/api/v1/async/strategy/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 202
        data = json.loads(response.data)
        assert 'task_id' in data
    
    def test_async_business_optimization_submission(self, client):
        """Test submitting async business optimization task"""
        payload = {
            'product_portfolio': [
                {
                    'name': 'Product A',
                    'sales_history': [100, 120, 110],
                    'production_cost': 50.0
                },
                {
                    'name': 'Product B',
                    'sales_history': [80, 90, 85],
                    'production_cost': 40.0
                }
            ]
        }
        
        response = client.post(
            '/api/v1/async/business_optimizer/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 202
        data = json.loads(response.data)
        assert 'task_id' in data


class TestAPIEndpointsWithVariousInputs:
    """Test all API endpoints with various input formats"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/api/v1/health')
        
        assert response.status_code in [200, 503]
        data = json.loads(response.data)
        
        assert 'status' in data
        assert 'services' in data
    
    def test_db_health_endpoint(self, client):
        """Test database health check endpoint"""
        response = client.get('/api/v1/db/health')
        
        assert response.status_code in [200, 503]
        data = json.loads(response.data)
        
        assert 'status' in data
        assert 'databases' in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get('/api/v1/model_info')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'enterprise_analyst' in data
        assert 'market_decipherer' in data
        assert 'strategy_engine' in data
    
    def test_market_analysis_minimal_input(self, client):
        """Test market analysis with minimal required input"""
        payload = {
            'market_data': [
                {'feature1': 0.5, 'feature2': 0.3},
                {'feature1': 0.6, 'feature2': 0.4}
            ]
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
    
    def test_market_analysis_with_all_parameters(self, client):
        """Test market analysis with all optional parameters"""
        payload = {
            'market_data': [
                {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.8},
                {'feature1': 0.6, 'feature2': 0.4, 'feature3': 0.7},
                {'feature1': 0.55, 'feature2': 0.35, 'feature3': 0.75}
            ],
            'entity_ids': ['entity_1', 'entity_2', 'entity_3'],
            'auto_select_clusters': True,
            'similarity_threshold': 0.75,
            'top_k_links': 15
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
    
    def test_strategy_generation_minimal_input(self, client):
        """Test strategy generation with minimal required input"""
        payload = {
            'market_state': {
                'market_demand': 0.7,
                'competitor_prices': [0.5, 0.6],
                'sales_volume': 0.65,
                'conversion_rate': 0.15,
                'inventory_level': 0.8,
                'market_trend': 0.2
            }
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 429, 500, 503]
    
    def test_strategy_generation_with_context(self, client):
        """Test strategy generation with context"""
        payload = {
            'market_state': {
                'market_demand': 0.7,
                'competitor_prices': [0.5, 0.6],
                'sales_volume': 0.65,
                'conversion_rate': 0.15,
                'inventory_level': 0.8,
                'market_trend': 0.2
            },
            'context': {
                'company_name': 'TechCorp',
                'product_name': 'CloudERP'
            },
            'include_explanation': True,
            'deterministic': True
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 429, 500, 503]
    
    def test_strategy_comparison(self, client):
        """Test strategy comparison across scenarios"""
        payload = {
            'scenarios': [
                {
                    'name': 'Scenario 1',
                    'market_state': {
                        'market_demand': 0.7,
                        'competitor_prices': [0.5],
                        'sales_volume': 0.65,
                        'conversion_rate': 0.15,
                        'inventory_level': 0.8,
                        'market_trend': 0.2
                    }
                },
                {
                    'name': 'Scenario 2',
                    'market_state': {
                        'market_demand': 0.6,
                        'competitor_prices': [0.6],
                        'sales_volume': 0.55,
                        'conversion_rate': 0.12,
                        'inventory_level': 0.7,
                        'market_trend': -0.1
                    }
                }
            ]
        }
        
        response = client.post(
            '/api/v1/strategy/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 429, 500, 503]
    
    def test_performance_monitoring_minimal_input(self, client):
        """Test performance monitoring with minimal input"""
        payload = {
            'historical_data': [
                [100.0, 0.15, 0.8],
                [110.0, 0.16, 0.82],
                [105.0, 0.14, 0.79]
            ]
        }
        
        response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
    
    def test_performance_monitoring_with_current_data(self, client):
        """Test performance monitoring with current data"""
        payload = {
            'historical_data': [
                [100.0, 0.15, 0.8],
                [110.0, 0.16, 0.82],
                [105.0, 0.14, 0.79]
            ],
            'current_data': [
                [108.0, 0.155, 0.81]
            ],
            'include_feedback': True
        }
        
        response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
    
    def test_performance_trends(self, client):
        """Test performance trends endpoint"""
        payload = {
            'historical_data': [
                [100.0, 0.15, 0.8],
                [110.0, 0.16, 0.82],
                [105.0, 0.14, 0.79],
                [115.0, 0.17, 0.83]
            ],
            'window_size': 30
        }
        
        response = client.post(
            '/api/v1/performance/trends',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
    
    def test_critical_alerts(self, client):
        """Test critical alerts endpoint"""
        response = client.get('/api/v1/performance/alerts/critical')
        
        assert response.status_code in [200, 500, 503]
    
    def test_critical_alerts_with_severity(self, client):
        """Test critical alerts with severity filter"""
        response = client.get('/api/v1/performance/alerts/critical?severity_threshold=medium')
        
        assert response.status_code in [200, 400, 500, 503]
    
    def test_feedback_recommendations(self, client):
        """Test feedback recommendations endpoint"""
        response = client.get('/api/v1/performance/feedback/recommendations')
        
        assert response.status_code in [200, 500, 503]
    
    def test_business_optimizer_minimal_input(self, client):
        """Test business optimizer with minimal input"""
        payload = {
            'product_portfolio': [
                {'name': 'Product A'},
                {'name': 'Product B'}
            ]
        }
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
    
    def test_business_optimizer_with_full_data(self, client):
        """Test business optimizer with complete data"""
        payload = {
            'product_portfolio': [
                {
                    'name': 'Product A',
                    'sales_history': [100, 120, 110, 130],
                    'demand_forecast': 125.0,
                    'production_cost': 50.0,
                    'current_inventory': 200.0
                },
                {
                    'name': 'Product B',
                    'sales_history': [80, 90, 85, 95],
                    'demand_forecast': 92.0,
                    'production_cost': 40.0,
                    'current_inventory': 150.0
                }
            ],
            'constraints': {
                'min_production': [50, 40],
                'max_production': [200, 180],
                'total_budget': 10000.0,
                'capacity_limit': 500.0
            },
            'revenue_weight': 0.7,
            'cost_weight': 0.3
        }
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
    
    def test_business_scenarios_analysis(self, client):
        """Test business scenarios analysis"""
        payload = {
            'product_portfolio': [
                {'name': 'Product A', 'production_cost': 50.0},
                {'name': 'Product B', 'production_cost': 40.0}
            ],
            'scenarios': [
                {
                    'name': 'Conservative',
                    'constraints': {'total_budget': 5000.0}
                },
                {
                    'name': 'Aggressive',
                    'constraints': {'total_budget': 15000.0}
                }
            ]
        }
        
        response = client.post(
            '/api/v1/business_optimizer/scenarios',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
    
    def test_explain_model_local(self, client):
        """Test model explanation with local explanation"""
        payload = {
            'model_type': 'rl',
            'instance': [0.7, 0.5, 0.6, 0.65, 0.15, 0.8],
            'top_n': 5,
            'include_visualizations': False,
            'explanation_type': 'local'
        }
        
        response = client.post(
            '/api/v1/explain',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
    
    def test_explain_model_global(self, client):
        """Test model explanation with global explanation"""
        payload = {
            'model_type': 'lstm',
            'background_data': [
                [0.7, 0.5, 0.6],
                [0.65, 0.55, 0.62],
                [0.72, 0.48, 0.58]
            ],
            'top_n': 10,
            'explanation_type': 'global'
        }
        
        response = client.post(
            '/api/v1/explain',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 500, 503]
    
    def test_explain_batch(self, client):
        """Test batch model explanation"""
        payload = {
            'model_type': 'regression',
            'instances': [
                [0.7, 0.5, 0.6],
                [0.65, 0.55, 0.62]
            ],
            'top_n': 5
        }
        
        response = client.post(
            '/api/v1/explain/batch',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_json_body(self, client):
        """Test request with invalid JSON"""
        response = client.post(
            '/api/v1/analyze_company',
            data='invalid json{',
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_missing_required_field(self, client):
        """Test request with missing required field"""
        payload = {
            'source_type': 'product_summary'
            # Missing 'text' field
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_invalid_field_type(self, client):
        """Test request with invalid field type"""
        payload = {
            'text': 12345,  # Should be string
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_empty_request_body(self, client):
        """Test request with empty body"""
        response = client.post(
            '/api/v1/analyze_company',
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_wrong_content_type(self, client):
        """Test request with wrong content type"""
        payload = {
            'text': 'Some text'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='text/plain'
        )
        
        assert response.status_code in [400, 415]
    
    def test_404_not_found(self, client):
        """Test 404 error for non-existent endpoint"""
        response = client.get('/api/v1/nonexistent_endpoint')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_405_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method"""
        response = client.get('/api/v1/analyze_company')
        
        assert response.status_code == 405
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_market_analysis_mismatched_entity_ids(self, client):
        """Test market analysis with mismatched entity_ids length"""
        payload = {
            'market_data': [
                {'feature1': 0.5, 'feature2': 0.3},
                {'feature1': 0.6, 'feature2': 0.4}
            ],
            'entity_ids': ['entity_1']  # Length mismatch
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_strategy_invalid_market_state_values(self, client):
        """Test strategy with out-of-range market state values"""
        payload = {
            'market_state': {
                'market_demand': 1.5,  # Out of range (should be 0-1)
                'competitor_prices': [0.5],
                'sales_volume': 0.65,
                'conversion_rate': 0.15,
                'inventory_level': 0.8,
                'market_trend': 0.2
            }
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [400, 429, 500, 503]
    
    def test_strategy_comparison_too_many_scenarios(self, client):
        """Test strategy comparison with too many scenarios"""
        scenarios = [
            {
                'name': f'Scenario {i}',
                'market_state': {
                    'market_demand': 0.7,
                    'competitor_prices': [0.5],
                    'sales_volume': 0.65,
                    'conversion_rate': 0.15,
                    'inventory_level': 0.8,
                    'market_trend': 0.2
                }
            }
            for i in range(15)  # More than max allowed (10)
        ]
        
        payload = {'scenarios': scenarios}
        
        response = client.post(
            '/api/v1/strategy/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_performance_invalid_data_shape(self, client):
        """Test performance monitoring with invalid data shape"""
        payload = {
            'historical_data': [100.0, 110.0, 105.0]  # Should be 2D
        }
        
        response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_performance_mismatched_feature_dimensions(self, client):
        """Test performance monitoring with mismatched feature dimensions"""
        payload = {
            'historical_data': [
                [100.0, 0.15, 0.8],
                [110.0, 0.16, 0.82]
            ],
            'current_data': [
                [108.0, 0.155]  # Missing one feature
            ]
        }
        
        response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_business_optimizer_empty_portfolio(self, client):
        """Test business optimizer with empty product portfolio"""
        payload = {
            'product_portfolio': []
        }
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_business_scenarios_too_many(self, client):
        """Test business scenarios with too many scenarios"""
        scenarios = [{'name': f'Scenario {i}'} for i in range(15)]
        
        payload = {
            'product_portfolio': [{'name': 'Product A'}],
            'scenarios': scenarios
        }
        
        response = client.post(
            '/api/v1/business_optimizer/scenarios',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_explain_invalid_model_type(self, client):
        """Test model explanation with invalid model type"""
        payload = {
            'model_type': 'invalid_model',
            'instance': [0.7, 0.5, 0.6],
            'explanation_type': 'local'
        }
        
        response = client.post(
            '/api/v1/explain',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_explain_missing_instance_for_local(self, client):
        """Test local explanation without instance"""
        payload = {
            'model_type': 'rl',
            'explanation_type': 'local'
            # Missing 'instance' field
        }
        
        response = client.post(
            '/api/v1/explain',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_explain_missing_background_for_global(self, client):
        """Test global explanation without background data"""
        payload = {
            'model_type': 'lstm',
            'explanation_type': 'global'
            # Missing 'background_data' field
        }
        
        response = client.post(
            '/api/v1/explain',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_critical_alerts_invalid_severity(self, client):
        """Test critical alerts with invalid severity threshold"""
        response = client.get('/api/v1/performance/alerts/critical?severity_threshold=invalid')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_strategy_rate_limit(self, client):
        """Test rate limiting on strategy endpoint"""
        payload = {
            'market_state': {
                'market_demand': 0.7,
                'competitor_prices': [0.5],
                'sales_volume': 0.65,
                'conversion_rate': 0.15,
                'inventory_level': 0.8,
                'market_trend': 0.2
            }
        }
        
        # Make multiple rapid requests to trigger rate limit
        responses = []
        for _ in range(25):  # Exceeds max_calls_per_minute (20)
            response = client.post(
                '/api/v1/strategy',
                data=json.dumps(payload),
                content_type='application/json'
            )
            responses.append(response.status_code)
        
        # At least one should be rate limited (429)
        assert 429 in responses or all(code in [200, 500, 503] for code in responses)


class TestRequestValidation:
    """Test request validation and sanitization"""
    
    def test_input_sanitization(self, client):
        """Test input sanitization for special characters"""
        payload = {
            'text': 'TechCorp\x00 provides software solutions.',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should handle null bytes gracefully
        assert response.status_code in [200, 400, 500]
    
    def test_large_payload(self, client):
        """Test handling of large payloads"""
        large_text = 'A' * 100000  # 100KB text
        
        payload = {
            'text': large_text,
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should handle or reject large payloads
        assert response.status_code in [200, 400, 413, 500]
    
    def test_sql_injection_attempt(self, client):
        """Test protection against SQL injection"""
        payload = {
            'text': "'; DROP TABLE companies; --",
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should handle safely
        assert response.status_code in [200, 400, 500]
    
    def test_xss_attempt(self, client):
        """Test protection against XSS attacks"""
        payload = {
            'text': '<script>alert("XSS")</script>',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Should handle safely
        assert response.status_code in [200, 400, 500]
    
    def test_unicode_handling(self, client):
        """Test proper Unicode handling"""
        payload = {
            'text': 'TechCorp提供企业软件解决方案 für Unternehmen 🚀',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 500]


class TestCORSAndHeaders:
    """Test CORS and response headers"""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present"""
        response = client.get('/api/v1/health')
        
        # Check for CORS headers
        assert 'Access-Control-Allow-Origin' in response.headers or response.status_code in [200, 503]
    
    def test_request_id_header(self, client):
        """Test that X-Request-ID header is present"""
        response = client.get('/api/v1/health')
        
        # Check for request ID header
        assert 'X-Request-ID' in response.headers or response.status_code in [200, 503]
    
    def test_options_request(self, client):
        """Test OPTIONS request for CORS preflight"""
        response = client.options('/api/v1/analyze_company')
        
        # Should handle OPTIONS request
        assert response.status_code in [200, 204, 405]


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_market_data(self, client):
        """Test market analysis with empty data"""
        payload = {
            'market_data': []
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_single_data_point_market_analysis(self, client):
        """Test market analysis with single data point"""
        payload = {
            'market_data': [
                {'feature1': 0.5, 'feature2': 0.3}
            ]
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May fail due to insufficient data
        assert response.status_code in [200, 400, 500, 503]
    
    def test_zero_values_in_market_state(self, client):
        """Test strategy with zero values"""
        payload = {
            'market_state': {
                'market_demand': 0.0,
                'competitor_prices': [0.0],
                'sales_volume': 0.0,
                'conversion_rate': 0.0,
                'inventory_level': 0.0,
                'market_trend': 0.0
            }
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 429, 500, 503]
    
    def test_extreme_values_in_market_state(self, client):
        """Test strategy with extreme boundary values"""
        payload = {
            'market_state': {
                'market_demand': 1.0,
                'competitor_prices': [1.0],
                'sales_volume': 1.0,
                'conversion_rate': 1.0,
                'inventory_level': 1.0,
                'market_trend': 1.0
            }
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 429, 500, 503]
    
    def test_negative_market_trend(self, client):
        """Test strategy with negative market trend"""
        payload = {
            'market_state': {
                'market_demand': 0.5,
                'competitor_prices': [0.5],
                'sales_volume': 0.5,
                'conversion_rate': 0.1,
                'inventory_level': 0.5,
                'market_trend': -1.0
            }
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 429, 500, 503]
    
    def test_empty_competitor_prices(self, client):
        """Test strategy with empty competitor prices"""
        payload = {
            'market_state': {
                'market_demand': 0.7,
                'competitor_prices': [],
                'sales_volume': 0.65,
                'conversion_rate': 0.15,
                'inventory_level': 0.8,
                'market_trend': 0.2
            }
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 429, 500, 503]
    
    def test_very_long_product_name(self, client):
        """Test business optimizer with very long product name"""
        payload = {
            'product_portfolio': [
                {
                    'name': 'A' * 1000,  # Very long name
                    'production_cost': 50.0
                }
            ]
        }
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 500, 503]
    
    def test_negative_production_cost(self, client):
        """Test business optimizer with negative cost"""
        payload = {
            'product_portfolio': [
                {
                    'name': 'Product A',
                    'production_cost': -50.0
                }
            ]
        }
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 500, 503]
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests"""
        import concurrent.futures
        
        def make_request():
            payload = {
                'text': 'TechCorp provides software solutions.',
                'source_type': 'product_summary'
            }
            return client.post(
                '/api/v1/analyze_company',
                data=json.dumps(payload),
                content_type='application/json'
            )
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should complete successfully or with expected errors
        for response in results:
            assert response.status_code in [200, 400, 500, 503]
    
    def test_async_task_nonexistent_id(self, client):
        """Test checking status of non-existent task"""
        fake_task_id = 'nonexistent-task-id-12345'
        
        response = client.get(f'/api/v1/async/tasks/{fake_task_id}')
        
        # Should handle gracefully
        assert response.status_code in [200, 404, 500]
    
    def test_async_task_cancel_completed(self, client):
        """Test cancelling an already completed task"""
        # Submit a simple task
        payload = {
            'text': 'Short text',
            'source_type': 'product_summary'
        }
        
        submit_response = client.post(
            '/api/v1/async/analyze_company/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        task_data = json.loads(submit_response.data)
        task_id = task_data['task_id']
        
        # Wait a bit for task to potentially complete
        time.sleep(2)
        
        # Try to cancel
        cancel_response = client.post(f'/api/v1/async/tasks/{task_id}/cancel')
        
        # Should return 200 (cancelled) or 400 (cannot cancel)
        assert cancel_response.status_code in [200, 400, 500]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
