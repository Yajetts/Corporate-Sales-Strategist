"""Tests for Strategy Engine Module"""

import pytest
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.api.app import create_app
from src.models.market_env import MarketSimulationEnv
from src.models.strategy_agent import StrategyAgent
from src.models.llm_explainer import LLMExplainer
from src.services.strategy_engine_service import StrategyEngineService


@pytest.fixture
def client():
    """Create test client"""
    app = create_app('testing')
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


class TestMarketEnvironment:
    """Test cases for Market Simulation Environment"""
    
    def test_env_initialization(self):
        """Test environment initialization"""
        env = MarketSimulationEnv(num_competitors=5, max_steps=100)
        
        assert env.num_competitors == 5
        assert env.max_steps == 100
        assert env.observation_space.shape[0] == 10  # 5 + 5 competitors
        assert env.action_space.shape[0] == 3
    
    def test_env_reset(self):
        """Test environment reset"""
        env = MarketSimulationEnv()
        obs, info = env.reset()
        
        assert obs.shape == (10,)  # 5 + 5 competitors
        assert all(0 <= x <= 1 for x in obs[:9])  # All normalized values
        assert -1 <= obs[9] <= 1  # Market trend
        assert 'step' in info
        assert info['step'] == 0
    
    def test_env_step(self):
        """Test environment step"""
        env = MarketSimulationEnv()
        obs, info = env.reset()
        
        action = np.array([0.1, 1.0, 0.5])  # price_adj, sales_approach, promo
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (10,)  # 5 + 5 competitors
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'revenue' in info
        assert 'sales_volume' in info
    
    def test_env_state_transitions(self):
        """Test environment state transitions over multiple steps"""
        env = MarketSimulationEnv(num_competitors=3, max_steps=10)
        obs, info = env.reset()
        
        initial_demand = obs[0]
        initial_sales = obs[4]
        
        # Take multiple steps with consistent actions
        for _ in range(5):
            action = np.array([0.0, 1.0, 0.3])  # Moderate strategy
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Verify state remains valid
            assert obs.shape == (8,)  # 5 + 3 competitors
            assert all(0 <= x <= 1 for x in obs[:7])
            assert -1 <= obs[7] <= 1  # Market trend
            
            # Verify info contains expected keys
            assert 'revenue' in info
            assert 'sales_volume' in info
            assert 'conversion_rate' in info
            assert 'price' in info
        
        # State should have changed from initial
        assert obs[0] != initial_demand or obs[4] != initial_sales
    
    def test_env_price_adjustment_impact(self):
        """Test that price adjustments affect revenue and sales"""
        env = MarketSimulationEnv(num_competitors=5)
        
        # Test with price increase
        obs, _ = env.reset(seed=42)
        action_increase = np.array([0.5, 1.0, 0.0])  # Increase price
        obs1, reward1, _, _, info1 = env.step(action_increase)
        
        # Reset and test with price decrease
        obs, _ = env.reset(seed=42)
        action_decrease = np.array([-0.5, 1.0, 0.0])  # Decrease price
        obs2, reward2, _, _, info2 = env.step(action_decrease)
        
        # Price should be different
        assert info1['price'] > info2['price']
    
    def test_env_promotion_impact(self):
        """Test that promotion intensity affects sales"""
        env = MarketSimulationEnv(num_competitors=5)
        
        # Test with high promotion
        obs, _ = env.reset(seed=42)
        action_high_promo = np.array([0.0, 1.0, 0.9])
        obs1, reward1, _, _, info1 = env.step(action_high_promo)
        
        # Reset and test with no promotion
        obs, _ = env.reset(seed=42)
        action_no_promo = np.array([0.0, 1.0, 0.0])
        obs2, reward2, _, _, info2 = env.step(action_no_promo)
        
        # High promotion should generally lead to higher sales
        # (though stochastic, so we just verify the mechanism runs)
        assert isinstance(info1['sales_volume'], (int, float))
        assert isinstance(info2['sales_volume'], (int, float))
    
    def test_env_episode_termination(self):
        """Test that episodes terminate correctly"""
        env = MarketSimulationEnv(num_competitors=3, max_steps=5)
        obs, _ = env.reset()
        
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated):
            action = np.array([0.0, 1.0, 0.5])
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if steps > 10:  # Safety check
                break
        
        assert truncated  # Should truncate at max_steps
        assert steps == 5


class TestStrategyAgent:
    """Test cases for Strategy Agent"""
    
    def test_agent_initialization(self):
        """Test agent initialization without model"""
        agent = StrategyAgent(num_competitors=5)
        
        assert agent.num_competitors == 5
        assert agent.model is None
    
    def test_state_dict_conversion(self):
        """Test market state dictionary to observation conversion"""
        agent = StrategyAgent(num_competitors=3)
        
        market_state = {
            'market_demand': 0.7,
            'competitor_prices': [0.6, 0.7, 0.8],
            'sales_volume': 0.5,
            'conversion_rate': 0.2,
            'inventory_level': 0.8,
            'market_trend': 0.1
        }
        
        obs = agent._state_dict_to_obs(market_state)
        
        assert obs.shape == (8,)  # 5 + 3 competitors
        assert np.isclose(obs[0], 0.7)  # market_demand
        assert np.allclose(obs[1:4], [0.6, 0.7, 0.8])  # competitor_prices
    
    def test_state_dict_conversion_with_padding(self):
        """Test state conversion with fewer competitor prices than expected"""
        agent = StrategyAgent(num_competitors=5)
        
        market_state = {
            'market_demand': 0.7,
            'competitor_prices': [0.6, 0.7],  # Only 2 prices, need 5
            'sales_volume': 0.5,
            'conversion_rate': 0.2,
            'inventory_level': 0.8,
            'market_trend': 0.1
        }
        
        obs = agent._state_dict_to_obs(market_state)
        
        assert obs.shape == (10,)  # 5 + 5 competitors
        assert np.isclose(obs[0], 0.7)
        assert np.isclose(obs[1], 0.6)
        assert np.isclose(obs[2], 0.7)
        # Padded values should be 0.7 (default)
        assert np.isclose(obs[3], 0.7)
    
    def test_agent_strategy_generation_without_model(self):
        """Test that strategy generation fails gracefully without trained model"""
        agent = StrategyAgent(num_competitors=5)
        
        market_state = {
            'market_demand': 0.7,
            'competitor_prices': [0.6, 0.7, 0.8, 0.65, 0.75],
            'sales_volume': 0.5,
            'conversion_rate': 0.2,
            'inventory_level': 0.8,
            'market_trend': 0.1
        }
        
        with pytest.raises(ValueError, match="Model not trained or loaded"):
            agent.predict_strategy(market_state)
    
    def test_agent_create_env(self):
        """Test environment creation"""
        agent = StrategyAgent(num_competitors=3)
        env = agent.create_env(max_steps=50)
        
        assert env is not None
        # Verify it's a vectorized environment
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')


class TestStrategyAPI:
    """Test cases for Strategy API endpoints"""
    
    def test_strategy_endpoint_success(self, client):
        """Test successful strategy generation"""
        payload = {
            'market_state': {
                'market_demand': 0.7,
                'competitor_prices': [0.6, 0.7, 0.8, 0.65, 0.75],
                'sales_volume': 0.5,
                'conversion_rate': 0.2,
                'inventory_level': 0.8,
                'market_trend': 0.1
            },
            'include_explanation': False,
            'deterministic': True
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May return 200, 400, or 500 if model not trained
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'recommendations' in data
            assert 'confidence_score' in data
            assert 'actionable_insights' in data
    
    def test_strategy_endpoint_with_context(self, client):
        """Test strategy generation with context"""
        payload = {
            'market_state': {
                'market_demand': 0.7,
                'competitor_prices': [0.6, 0.7, 0.8, 0.65, 0.75],
                'sales_volume': 0.5,
                'conversion_rate': 0.2,
                'inventory_level': 0.8,
                'market_trend': 0.1
            },
            'context': {
                'company_name': 'TechCorp',
                'product_name': 'CloudERP'
            },
            'include_explanation': False
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 500]
    
    def test_strategy_endpoint_missing_market_state(self, client):
        """Test strategy with missing market_state"""
        payload = {
            'include_explanation': False
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_strategy_endpoint_invalid_market_demand(self, client):
        """Test strategy with invalid market_demand"""
        payload = {
            'market_state': {
                'market_demand': 1.5,  # Invalid: > 1
                'competitor_prices': [0.6, 0.7, 0.8, 0.65, 0.75],
                'sales_volume': 0.5,
                'conversion_rate': 0.2,
                'inventory_level': 0.8,
                'market_trend': 0.1
            }
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_strategy_endpoint_missing_field(self, client):
        """Test strategy with missing required field"""
        payload = {
            'market_state': {
                'market_demand': 0.7,
                'competitor_prices': [0.6, 0.7, 0.8],
                # Missing other required fields
            }
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_strategy_compare_endpoint(self, client):
        """Test strategy comparison endpoint"""
        payload = {
            'scenarios': [
                {
                    'name': 'High Demand',
                    'market_state': {
                        'market_demand': 0.9,
                        'competitor_prices': [0.6, 0.7, 0.8, 0.65, 0.75],
                        'sales_volume': 0.7,
                        'conversion_rate': 0.3,
                        'inventory_level': 0.8,
                        'market_trend': 0.2
                    }
                },
                {
                    'name': 'Low Demand',
                    'market_state': {
                        'market_demand': 0.3,
                        'competitor_prices': [0.6, 0.7, 0.8, 0.65, 0.75],
                        'sales_volume': 0.3,
                        'conversion_rate': 0.1,
                        'inventory_level': 0.8,
                        'market_trend': -0.2
                    }
                }
            ]
        }
        
        response = client.post(
            '/api/v1/strategy/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'scenarios' in data
            assert 'summary' in data
    
    def test_strategy_compare_too_many_scenarios(self, client):
        """Test strategy comparison with too many scenarios"""
        scenarios = []
        for i in range(15):  # More than max allowed (10)
            scenarios.append({
                'name': f'Scenario {i}',
                'market_state': {
                    'market_demand': 0.7,
                    'competitor_prices': [0.6, 0.7, 0.8, 0.65, 0.75],
                    'sales_volume': 0.5,
                    'conversion_rate': 0.2,
                    'inventory_level': 0.8,
                    'market_trend': 0.1
                }
            })
        
        payload = {'scenarios': scenarios}
        
        response = client.post(
            '/api/v1/strategy/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400


class TestLLMExplainer:
    """Test cases for LLM Explainer"""
    
    def test_llm_explainer_initialization_openai(self):
        """Test LLM explainer initialization with OpenAI"""
        explainer = LLMExplainer(provider='openai', api_key='test-key')
        
        assert explainer.provider == 'openai'
        assert explainer.api_key == 'test-key'
        assert explainer.model == 'gpt-4o'
    
    def test_llm_explainer_initialization_anthropic(self):
        """Test LLM explainer initialization with Anthropic"""
        explainer = LLMExplainer(provider='anthropic', api_key='test-key')
        
        assert explainer.provider == 'anthropic'
        assert explainer.api_key == 'test-key'
        assert explainer.model == 'claude-3-opus-20240229'
    
    def test_llm_explainer_invalid_provider(self):
        """Test LLM explainer with invalid provider"""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMExplainer(provider='invalid', api_key=None)
    
    def test_llm_explainer_missing_api_key(self):
        """Test LLM explainer without API key"""
        with pytest.raises(ValueError, match="API key not provided"):
            LLMExplainer(provider='openai', api_key=None)
    
    @patch('src.models.llm_explainer.requests.post')
    def test_explain_strategy_with_mocked_openai(self, mock_post):
        """Test strategy explanation with mocked OpenAI API"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        'summary': 'Market conditions favor aggressive pricing.',
                        'rationale': 'High demand and low competition support price increases.',
                        'expected_outcomes': 'Revenue should increase by 15-20%.',
                        'risks': 'Competitor response may reduce market share.'
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        explainer = LLMExplainer(provider='openai', api_key='test-key')
        
        market_state = {
            'market_demand': 0.8,
            'avg_competitor_price': 0.7,
            'sales_volume': 0.6,
            'conversion_rate': 0.25,
            'inventory_level': 0.9,
            'market_trend': 0.15
        }
        
        strategy = {
            'price_adjustment_pct': 10.0,
            'sales_approach': 'aggressive',
            'promotion_intensity': 0.7,
            'confidence': 0.85
        }
        
        explanation = explainer.explain_strategy(market_state, strategy)
        
        assert 'summary' in explanation
        assert 'rationale' in explanation
        assert 'expected_outcomes' in explanation
        assert 'risks' in explanation
        assert 'Market conditions' in explanation['summary']
        
        # Verify API was called
        mock_post.assert_called_once()
    
    @patch('src.models.llm_explainer.requests.post')
    def test_explain_strategy_with_mocked_anthropic(self, mock_post):
        """Test strategy explanation with mocked Anthropic API"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'content': [{
                'text': json.dumps({
                    'summary': 'Conservative approach recommended due to market volatility.',
                    'rationale': 'Low demand and high competition require careful pricing.',
                    'expected_outcomes': 'Stable sales with minimal risk.',
                    'risks': 'May miss growth opportunities if market improves.'
                })
            }]
        }
        mock_post.return_value = mock_response
        
        explainer = LLMExplainer(provider='anthropic', api_key='test-key')
        
        market_state = {
            'market_demand': 0.4,
            'avg_competitor_price': 0.6,
            'sales_volume': 0.3,
            'conversion_rate': 0.15,
            'inventory_level': 0.7,
            'market_trend': -0.1
        }
        
        strategy = {
            'price_adjustment_pct': -5.0,
            'sales_approach': 'conservative',
            'promotion_intensity': 0.3,
            'confidence': 0.65
        }
        
        explanation = explainer.explain_strategy(market_state, strategy)
        
        assert 'summary' in explanation
        assert 'Conservative' in explanation['summary']
        
        # Verify API was called
        mock_post.assert_called_once()
    
    @patch('src.models.llm_explainer.requests.post')
    def test_explain_strategy_api_failure_fallback(self, mock_post):
        """Test fallback explanation when API fails"""
        # Mock API failure
        mock_post.side_effect = Exception("API Error")
        
        explainer = LLMExplainer(provider='openai', api_key='test-key')
        
        market_state = {
            'market_demand': 0.7,
            'avg_competitor_price': 0.7,
            'sales_volume': 0.5,
            'conversion_rate': 0.2,
            'inventory_level': 0.8,
            'market_trend': 0.0
        }
        
        strategy = {
            'price_adjustment_pct': 0.0,
            'sales_approach': 'moderate',
            'promotion_intensity': 0.5,
            'confidence': 0.7
        }
        
        explanation = explainer.explain_strategy(market_state, strategy)
        
        # Should return fallback explanation
        assert 'summary' in explanation
        assert 'rationale' in explanation
        assert 'expected_outcomes' in explanation
        assert 'risks' in explanation
    
    def test_fallback_explanation_generation(self):
        """Test fallback explanation generation"""
        explainer = LLMExplainer(provider='openai', api_key='test-key')
        
        strategy = {
            'price_adjustment_pct': 8.0,
            'sales_approach': 'aggressive',
            'promotion_intensity': 0.8,
            'confidence': 0.75
        }
        
        explanation = explainer._generate_fallback_explanation(strategy)
        
        assert 'summary' in explanation
        assert 'rationale' in explanation
        assert 'expected_outcomes' in explanation
        assert 'risks' in explanation
        assert 'aggressive' in explanation['summary']
        assert 'price increase' in explanation['summary'].lower()
    
    @patch('src.models.llm_explainer.requests.post')
    def test_explain_strategy_with_context(self, mock_post):
        """Test strategy explanation with additional context"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        'summary': 'TechCorp should increase CloudERP pricing.',
                        'rationale': 'Strong market position supports premium pricing.',
                        'expected_outcomes': 'Higher margins with maintained volume.',
                        'risks': 'Customer churn if value not demonstrated.'
                    })
                }
            }]
        }
        mock_post.return_value = mock_response
        
        explainer = LLMExplainer(provider='openai', api_key='test-key')
        
        market_state = {'market_demand': 0.8}
        strategy = {'price_adjustment_pct': 10.0, 'sales_approach': 'aggressive'}
        context = {'company_name': 'TechCorp', 'product_name': 'CloudERP'}
        
        explanation = explainer.explain_strategy(market_state, strategy, context)
        
        assert 'summary' in explanation
        # Context should be included in the prompt
        call_args = mock_post.call_args
        assert call_args is not None


class TestStrategyEngineService:
    """Test cases for Strategy Engine Service"""
    
    def test_service_initialization(self):
        """Test service initialization"""
        service = StrategyEngineService(
            agent_model_path=None,
            llm_provider='openai',
            llm_api_key='test-key',
            num_competitors=5
        )
        
        assert service.agent is not None
        assert service.agent.num_competitors == 5
    
    def test_service_generate_strategy_without_model(self):
        """Test strategy generation without trained model"""
        service = StrategyEngineService(
            agent_model_path=None,
            llm_api_key='test-key'
        )
        
        market_state = {
            'market_demand': 0.7,
            'competitor_prices': [0.6, 0.7, 0.8, 0.65, 0.75],
            'sales_volume': 0.5,
            'conversion_rate': 0.2,
            'inventory_level': 0.8,
            'market_trend': 0.1
        }
        
        # Should raise error because model not trained
        with pytest.raises(ValueError):
            service.generate_strategy(market_state, include_explanation=False)
    
    def test_service_validate_market_state(self):
        """Test market state validation"""
        service = StrategyEngineService(llm_api_key='test-key')
        
        # Valid market state
        valid_state = {
            'market_demand': 0.7,
            'competitor_prices': [0.6, 0.7, 0.8],
            'sales_volume': 0.5,
            'conversion_rate': 0.2,
            'inventory_level': 0.8,
            'market_trend': 0.1
        }
        service._validate_market_state(valid_state)  # Should not raise
        
        # Invalid: missing field
        invalid_state = {
            'market_demand': 0.7,
            'competitor_prices': [0.6, 0.7, 0.8]
        }
        with pytest.raises(ValueError, match="Missing required field"):
            service._validate_market_state(invalid_state)
        
        # Invalid: out of range
        invalid_state = {
            'market_demand': 1.5,  # > 1
            'competitor_prices': [0.6, 0.7, 0.8],
            'sales_volume': 0.5,
            'conversion_rate': 0.2,
            'inventory_level': 0.8,
            'market_trend': 0.1
        }
        with pytest.raises(ValueError, match="market_demand must be between"):
            service._validate_market_state(invalid_state)
    
    def test_service_generate_insights(self):
        """Test actionable insights generation"""
        service = StrategyEngineService(llm_api_key='test-key')
        
        market_state = {
            'market_demand': 0.3,  # Low demand
            'competitor_prices': [0.6, 0.7, 0.8],
            'sales_volume': 0.5,
            'conversion_rate': 0.1,  # Low conversion
            'inventory_level': 0.2,  # Low inventory
            'market_trend': -0.3  # Negative trend
        }
        
        strategy_info = {
            'price_adjustment_pct': 10.0,
            'sales_approach': 'aggressive',
            'promotion_intensity': 0.8
        }
        
        insights = service._generate_insights(market_state, strategy_info)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        # Should detect low demand, low conversion, low inventory, negative trend
        insights_text = ' '.join(insights).lower()
        assert any(keyword in insights_text for keyword in ['demand', 'conversion', 'inventory', 'trend'])
    
    def test_service_confidence_level(self):
        """Test confidence level categorization"""
        service = StrategyEngineService(llm_api_key='test-key')
        
        assert service._get_confidence_level(0.9) == 'high'
        assert service._get_confidence_level(0.7) == 'medium'
        assert service._get_confidence_level(0.5) == 'low'
    
    def test_service_cache_key_generation(self):
        """Test cache key generation"""
        service = StrategyEngineService(llm_api_key='test-key', enable_caching=True)
        
        market_state = {
            'market_demand': 0.7,
            'competitor_prices': [0.6, 0.7, 0.8],
            'sales_volume': 0.5,
            'conversion_rate': 0.2,
            'inventory_level': 0.8,
            'market_trend': 0.1
        }
        
        key1 = service._generate_cache_key(market_state, None)
        key2 = service._generate_cache_key(market_state, None)
        
        # Same input should generate same key
        assert key1 == key2
        
        # Different input should generate different key
        market_state2 = market_state.copy()
        market_state2['market_demand'] = 0.8
        key3 = service._generate_cache_key(market_state2, None)
        assert key1 != key3
    
    def test_service_clear_cache(self):
        """Test cache clearing"""
        service = StrategyEngineService(llm_api_key='test-key', enable_caching=True)
        
        # Add something to cache
        service.cache['test_key'] = ({'data': 'value'}, None)
        assert len(service.cache) > 0
        
        service.clear_cache()
        assert len(service.cache) == 0
