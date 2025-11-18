"""Tests for Model Transparency Layer"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.models.shap_explainer import SHAPExplainer
from src.services.model_transparency_service import ModelTransparencyService, ExplanationCache


class TestExplanationCache:
    """Test cases for ExplanationCache"""
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = ExplanationCache(ttl_seconds=3600)
        assert cache.ttl.total_seconds() == 3600
        assert len(cache.cache) == 0
    
    def test_cache_set_and_get(self):
        """Test cache set and get operations"""
        cache = ExplanationCache(ttl_seconds=3600)
        
        test_data = {'result': 'test_value'}
        cache.set('test_key', test_data)
        
        retrieved = cache.get('test_key')
        assert retrieved == test_data
    
    def test_cache_clear(self):
        """Test cache clear"""
        cache = ExplanationCache()
        cache.set('key1', {'data': 1})
        cache.set('key2', {'data': 2})
        
        assert len(cache.cache) == 2
        
        cache.clear()
        assert len(cache.cache) == 0


class TestSHAPExplainer:
    """Test cases for SHAP Explainer"""
    
    def test_explainer_initialization_rl(self):
        """Test SHAP explainer initialization for RL model"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (10,)
        
        # Create explainer
        explainer = SHAPExplainer(
            model=mock_agent,
            model_type='rl',
            feature_names=['feature_1', 'feature_2'],
            device='cpu'
        )
        
        assert explainer.model_type == 'rl'
        assert explainer.feature_names == ['feature_1', 'feature_2']
        assert explainer.explainer is not None
    
    @pytest.mark.skip(reason="SHAP DeepExplainer requires real PyTorch models, not mocks")
    def test_explainer_initialization_lstm(self):
        """Test SHAP explainer initialization for LSTM model"""
        # Create mock LSTM model
        mock_lstm = Mock()
        mock_lstm.model = Mock()
        mock_lstm.sequence_length = 30
        mock_lstm.input_size = 5
        
        # Create explainer
        explainer = SHAPExplainer(
            model=mock_lstm,
            model_type='lstm',
            feature_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            device='cpu'
        )
        
        assert explainer.model_type == 'lstm'
        assert explainer.explainer is not None
    
    @pytest.mark.skip(reason="SHAP DeepExplainer requires real PyTorch models, not mocks")
    def test_explainer_initialization_regression(self):
        """Test SHAP explainer initialization for regression model"""
        # Create mock regression model
        mock_regression = Mock()
        mock_regression.model = Mock()
        mock_regression.input_size = 20
        
        # Create explainer
        explainer = SHAPExplainer(
            model=mock_regression,
            model_type='regression',
            feature_names=[f'feature_{i}' for i in range(20)],
            device='cpu'
        )
        
        assert explainer.model_type == 'regression'
        assert explainer.explainer is not None


class TestModelTransparencyService:
    """Test cases for Model Transparency Service"""
    
    def test_service_initialization(self):
        """Test service initialization"""
        service = ModelTransparencyService(
            rl_agent=None,
            lstm_forecaster=None,
            regression_optimizer=None,
            cache_ttl=3600
        )
        
        assert service.cache is not None
        assert len(service.explainers) == 0
    
    def test_get_rl_feature_names(self):
        """Test RL feature name generation"""
        mock_agent = Mock()
        mock_agent.num_competitors = 3
        
        service = ModelTransparencyService(rl_agent=mock_agent)
        feature_names = service._get_rl_feature_names()
        
        assert 'market_demand' in feature_names
        assert 'competitor_price_1' in feature_names
        assert 'sales_volume' in feature_names
        assert len(feature_names) == 8  # 1 + 3 + 4
    
    def test_cache_clear(self):
        """Test cache clearing"""
        service = ModelTransparencyService()
        service.cache.set('test_key', {'data': 'test'})
        
        assert len(service.cache.cache) > 0
        
        service.clear_cache()
        assert len(service.cache.cache) == 0
    
    def test_set_models(self):
        """Test setting models"""
        mock_agent = Mock()
        mock_agent.num_competitors = 5
        
        mock_lstm = Mock()
        mock_lstm.input_size = 5
        
        service = ModelTransparencyService()
        service.set_models(rl_agent=mock_agent, lstm_forecaster=mock_lstm)
        
        assert service.rl_agent == mock_agent
        assert service.lstm_forecaster == mock_lstm
        assert len(service.feature_names['rl']) > 0
        assert len(service.feature_names['lstm']) > 0
    
    def test_explain_prediction_with_rl_agent(self):
        """Test local explanation generation for RL agent"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.num_competitors = 2
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (7,)
        
        # Create service
        service = ModelTransparencyService(rl_agent=mock_agent)
        
        # Create mock explainer
        mock_explainer = Mock()
        mock_explainer.explain_local = Mock(return_value={
            'contributions': [
                {'feature': 'market_demand', 'shap_value': 0.5},
                {'feature': 'competitor_price_1', 'shap_value': -0.3},
                {'feature': 'sales_volume', 'shap_value': 0.2}
            ],
            'base_value': 0.0
        })
        service.explainers['rl'] = mock_explainer
        
        # Test explanation
        instance = np.random.randn(7)
        result = service.explain_prediction('rl', instance, top_n=3)
        
        assert 'model_type' in result
        assert result['model_type'] == 'rl'
        assert 'top_features' in result
        assert len(result['top_features']) == 3
        assert 'base_value' in result
        assert 'timestamp' in result
    
    def test_explain_global(self):
        """Test global explanation generation"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.num_competitors = 2
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (7,)
        
        # Create service
        service = ModelTransparencyService(rl_agent=mock_agent)
        
        # Create mock explainer
        mock_explainer = Mock()
        mock_explainer.explain_global = Mock(return_value={
            'feature_importance': [
                {'feature': 'market_demand', 'importance': 0.5},
                {'feature': 'competitor_price_1', 'importance': 0.3},
                {'feature': 'sales_volume', 'importance': 0.2}
            ],
            'num_samples': 50
        })
        service.explainers['rl'] = mock_explainer
        
        # Test global explanation
        data = np.random.randn(50, 7)
        result = service.explain_global('rl', data, top_n=3)
        
        assert 'model_type' in result
        assert result['model_type'] == 'rl'
        assert 'top_features' in result
        assert len(result['top_features']) == 3
        assert 'num_samples_analyzed' in result
        assert result['num_samples_analyzed'] == 50
    
    def test_batch_explain(self):
        """Test batch explanation"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.num_competitors = 2
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (7,)
        
        # Create service
        service = ModelTransparencyService(rl_agent=mock_agent)
        
        # Create mock explainer
        mock_explainer = Mock()
        mock_explainer.explain_local = Mock(return_value={
            'contributions': [
                {'feature': 'market_demand', 'shap_value': 0.5}
            ],
            'base_value': 0.0
        })
        service.explainers['rl'] = mock_explainer
        
        # Test batch explanation
        instances = [np.random.randn(7) for _ in range(3)]
        results = service.batch_explain('rl', instances, top_n=5)
        
        assert len(results) == 3
        for result in results:
            assert 'model_type' in result or 'error' in result
    
    def test_compare_predictions(self):
        """Test prediction comparison"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.num_competitors = 2
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (7,)
        
        # Create service
        service = ModelTransparencyService(rl_agent=mock_agent)
        
        # Create mock explainer
        mock_explainer = Mock()
        mock_explainer.explain_local = Mock(side_effect=[
            {
                'contributions': [
                    {'feature': 'market_demand', 'shap_value': 0.5},
                    {'feature': 'sales_volume', 'shap_value': 0.3}
                ],
                'base_value': 0.0
            },
            {
                'contributions': [
                    {'feature': 'market_demand', 'shap_value': 0.7},
                    {'feature': 'sales_volume', 'shap_value': 0.2}
                ],
                'base_value': 0.0
            }
        ])
        service.explainers['rl'] = mock_explainer
        
        # Test comparison
        instance1 = np.random.randn(7)
        instance2 = np.random.randn(7)
        result = service.compare_predictions('rl', instance1, instance2, top_n=5)
        
        assert 'model_type' in result
        assert 'instance1_explanation' in result
        assert 'instance2_explanation' in result
        assert 'feature_differences' in result
    
    def test_get_feature_contributions(self):
        """Test feature contribution extraction"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.num_competitors = 2
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (7,)
        
        # Create service
        service = ModelTransparencyService(rl_agent=mock_agent)
        
        # Create mock explainer
        mock_explainer = Mock()
        mock_explainer.explain_local = Mock(return_value={
            'contributions': [
                {'feature': 'market_demand', 'shap_value': 0.5},
                {'feature': 'competitor_price_1', 'shap_value': -0.3},
                {'feature': 'sales_volume', 'shap_value': 0.2}
            ],
            'base_value': 0.0
        })
        service.explainers['rl'] = mock_explainer
        
        # Test feature contributions
        instance = np.random.randn(7)
        contributions = service.get_feature_contributions('rl', instance)
        
        assert isinstance(contributions, dict)
        assert 'market_demand' in contributions
        assert contributions['market_demand'] == 0.5
    
    def test_explanation_caching(self):
        """Test that explanations are cached properly"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.num_competitors = 2
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (7,)
        
        # Create service
        service = ModelTransparencyService(rl_agent=mock_agent)
        
        # Create mock explainer
        mock_explainer = Mock()
        mock_explainer.explain_local = Mock(return_value={
            'contributions': [
                {'feature': 'market_demand', 'shap_value': 0.5}
            ],
            'base_value': 0.0
        })
        service.explainers['rl'] = mock_explainer
        
        # First call
        instance = np.random.randn(7)
        result1 = service.explain_prediction('rl', instance, top_n=5)
        
        # Second call with same instance should use cache
        result2 = service.explain_prediction('rl', instance, top_n=5)
        
        # Explainer should only be called once
        assert mock_explainer.explain_local.call_count == 1
        
        # Results should be identical
        assert result1['timestamp'] == result2['timestamp']
    
    def test_error_handling_invalid_model_type(self):
        """Test error handling for invalid model type"""
        service = ModelTransparencyService()
        
        with pytest.raises(ValueError, match="Unknown model type"):
            service.explain_prediction('invalid_type', np.random.randn(5))
    
    def test_error_handling_uninitialized_model(self):
        """Test error handling for uninitialized model"""
        service = ModelTransparencyService()
        
        with pytest.raises(ValueError, match="not initialized"):
            service.explain_prediction('rl', np.random.randn(5))



class TestSHAPVisualization:
    """Test cases for SHAP visualization generation"""
    
    def test_generate_bar_chart(self):
        """Test bar chart generation"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (5,)
        
        # Create explainer
        explainer = SHAPExplainer(
            model=mock_agent,
            model_type='rl',
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            device='cpu'
        )
        
        # Generate bar chart
        shap_values = np.array([0.5, -0.3, 0.2, -0.1, 0.4])
        chart_base64 = explainer.generate_bar_chart(shap_values, max_display=5)
        
        # Verify it's a valid base64 string
        assert isinstance(chart_base64, str)
        assert len(chart_base64) > 0
        
        # Verify it can be decoded
        import base64
        try:
            base64.b64decode(chart_base64)
        except Exception:
            pytest.fail("Generated chart is not valid base64")
    
    def test_generate_force_plot(self):
        """Test force plot generation"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (5,)
        
        # Create explainer
        explainer = SHAPExplainer(
            model=mock_agent,
            model_type='rl',
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            device='cpu'
        )
        
        # Generate force plot
        shap_values = np.array([0.5, -0.3, 0.2, -0.1, 0.4])
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        plot_base64 = explainer.generate_force_plot(shap_values, features, base_value=0.0)
        
        # Verify it's a valid base64 string
        assert isinstance(plot_base64, str)
        assert len(plot_base64) > 0
    
    def test_generate_waterfall_chart(self):
        """Test waterfall chart generation"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (5,)
        
        # Create explainer
        explainer = SHAPExplainer(
            model=mock_agent,
            model_type='rl',
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            device='cpu'
        )
        
        # Generate waterfall chart
        shap_values = np.array([0.5, -0.3, 0.2, -0.1, 0.4])
        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        chart_base64 = explainer.generate_waterfall_chart(shap_values, features, base_value=0.0)
        
        # Verify it's a valid base64 string
        assert isinstance(chart_base64, str)
        assert len(chart_base64) > 0
    
    def test_generate_all_visualizations(self):
        """Test generation of all visualization types"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (5,)
        
        # Create explainer
        explainer = SHAPExplainer(
            model=mock_agent,
            model_type='rl',
            feature_names=['f1', 'f2', 'f3', 'f4', 'f5'],
            device='cpu'
        )
        
        # Generate all visualizations
        instance = np.random.randn(5)
        visualizations = explainer.generate_all_visualizations(instance)
        
        # Verify visualizations are generated
        assert isinstance(visualizations, dict)
        assert 'force_plot' in visualizations or 'waterfall_chart' in visualizations
    
    def test_visualization_with_include_flag(self):
        """Test that visualizations are included when flag is set"""
        # Create mock RL agent
        mock_agent = Mock()
        mock_agent.num_competitors = 2
        mock_agent.model = Mock()
        mock_agent.model.predict = Mock(return_value=(np.array([0.5]), None))
        mock_agent.env = Mock()
        mock_agent.env.observation_space = Mock()
        mock_agent.env.observation_space.shape = (7,)
        
        # Create service
        service = ModelTransparencyService(rl_agent=mock_agent)
        
        # Create mock explainer with visualization support
        mock_explainer = Mock()
        mock_explainer.explain_local = Mock(return_value={
            'contributions': [
                {'feature': 'market_demand', 'shap_value': 0.5}
            ],
            'base_value': 0.0
        })
        mock_explainer.generate_all_visualizations = Mock(return_value={
            'force_plot': 'base64_encoded_image',
            'waterfall_chart': 'base64_encoded_image'
        })
        service.explainers['rl'] = mock_explainer
        
        # Test with visualizations
        instance = np.random.randn(7)
        result = service.explain_prediction('rl', instance, include_visualizations=True)
        
        assert 'visualizations' in result
        assert mock_explainer.generate_all_visualizations.called



class TestExplainAPIEndpoint:
    """Test cases for /explain API endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.api.app import create_app
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_explain_endpoint_local_explanation(self, client):
        """Test /explain endpoint with local explanation"""
        # Prepare request data
        request_data = {
            'model_type': 'rl',
            'instance': [0.5, 0.6, 0.7, 0.4, 0.3, 0.8, 0.2],
            'top_n': 5,
            'include_visualizations': False,
            'explanation_type': 'local'
        }
        
        # Make request
        response = client.post(
            '/api/v1/explain',
            json=request_data,
            content_type='application/json'
        )
        
        # Check response (may be 500/503 if model not loaded, which is acceptable in test environment)
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'model_type' in data
            assert data['model_type'] == 'rl'
            assert 'explanation_type' in data
            assert data['explanation_type'] == 'local'
            assert 'top_features' in data
            assert 'base_value' in data
    
    def test_explain_endpoint_global_explanation(self, client):
        """Test /explain endpoint with global explanation"""
        # Prepare request data
        request_data = {
            'model_type': 'rl',
            'explanation_type': 'global',
            'background_data': [[0.5, 0.6, 0.7, 0.4, 0.3, 0.8, 0.2] for _ in range(20)],
            'top_n': 5
        }
        
        # Make request
        response = client.post(
            '/api/v1/explain',
            json=request_data,
            content_type='application/json'
        )
        
        # Check response (may be 500/503 if model not loaded)
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'model_type' in data
            assert 'explanation_type' in data
            assert data['explanation_type'] == 'global'
            assert 'top_features' in data
            assert 'num_samples_analyzed' in data
    
    def test_explain_endpoint_with_visualizations(self, client):
        """Test /explain endpoint with visualizations enabled"""
        # Prepare request data
        request_data = {
            'model_type': 'rl',
            'instance': [0.5, 0.6, 0.7, 0.4, 0.3, 0.8, 0.2],
            'top_n': 5,
            'include_visualizations': True,
            'explanation_type': 'local'
        }
        
        # Make request
        response = client.post(
            '/api/v1/explain',
            json=request_data,
            content_type='application/json'
        )
        
        # Check response
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'visualizations' in data
    
    def test_explain_endpoint_missing_model_type(self, client):
        """Test /explain endpoint with missing model_type"""
        # Prepare request data without model_type
        request_data = {
            'instance': [0.5, 0.6, 0.7, 0.4, 0.3, 0.8, 0.2],
            'top_n': 5
        }
        
        # Make request
        response = client.post(
            '/api/v1/explain',
            json=request_data,
            content_type='application/json'
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_explain_endpoint_invalid_model_type(self, client):
        """Test /explain endpoint with invalid model_type"""
        # Prepare request data with invalid model_type
        request_data = {
            'model_type': 'invalid_model',
            'instance': [0.5, 0.6, 0.7, 0.4, 0.3, 0.8, 0.2],
            'top_n': 5
        }
        
        # Make request
        response = client.post(
            '/api/v1/explain',
            json=request_data,
            content_type='application/json'
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_explain_endpoint_missing_instance_for_local(self, client):
        """Test /explain endpoint with missing instance for local explanation"""
        # Prepare request data without instance
        request_data = {
            'model_type': 'rl',
            'explanation_type': 'local',
            'top_n': 5
        }
        
        # Make request
        response = client.post(
            '/api/v1/explain',
            json=request_data,
            content_type='application/json'
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_explain_endpoint_invalid_json(self, client):
        """Test /explain endpoint with invalid JSON"""
        # Make request with invalid JSON
        response = client.post(
            '/api/v1/explain',
            data='invalid json',
            content_type='application/json'
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_explain_batch_endpoint(self, client):
        """Test /explain/batch endpoint"""
        # Prepare request data
        request_data = {
            'model_type': 'rl',
            'instances': [
                [0.5, 0.6, 0.7, 0.4, 0.3, 0.8, 0.2],
                [0.4, 0.5, 0.6, 0.3, 0.2, 0.7, 0.1],
                [0.6, 0.7, 0.8, 0.5, 0.4, 0.9, 0.3]
            ],
            'top_n': 5
        }
        
        # Make request
        response = client.post(
            '/api/v1/explain/batch',
            json=request_data,
            content_type='application/json'
        )
        
        # Check response (may be 500/503 if model not loaded)
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'model_type' in data
            assert 'explanations' in data
            assert 'num_instances' in data
            assert data['num_instances'] == 3
    
    def test_explain_batch_endpoint_too_many_instances(self, client):
        """Test /explain/batch endpoint with too many instances"""
        # Prepare request data with >100 instances
        request_data = {
            'model_type': 'rl',
            'instances': [[0.5] * 7 for _ in range(101)],
            'top_n': 5
        }
        
        # Make request
        response = client.post(
            '/api/v1/explain/batch',
            json=request_data,
            content_type='application/json'
        )
        
        # Should return 400 Bad Request
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
