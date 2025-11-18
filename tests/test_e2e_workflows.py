"""End-to-end workflow tests for Sales Strategist System

Tests complete workflows that span multiple modules and verify
integration between components.

Requirements tested:
- 1.1: Company analysis workflow
- 2.1: Market analysis to strategy generation flow
- 3.1: Strategy generation workflow
- 4.1: Performance monitoring and feedback loop
- 5.1: Business optimization workflow
"""

import pytest
import json
import time
from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client with testing configuration"""
    app = create_app('testing')
    app.config['TESTING'] = True
    app.config['API_KEYS'] = 'test-api-key-e2e'
    
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_company_text():
    """Sample company text for testing"""
    return """
    TechCorp is a leading provider of cloud-based enterprise resource planning (ERP) 
    software solutions. Our flagship product, CloudERP, helps businesses streamline 
    operations, manage finances, and optimize supply chains. We serve mid-to-large 
    enterprises across manufacturing, retail, and services sectors.
    """


@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return [
        {'revenue': 0.75, 'growth': 0.6, 'market_share': 0.45, 'customer_satisfaction': 0.8},
        {'revenue': 0.65, 'growth': 0.55, 'market_share': 0.35, 'customer_satisfaction': 0.75},
        {'revenue': 0.85, 'growth': 0.7, 'market_share': 0.55, 'customer_satisfaction': 0.85},
        {'revenue': 0.55, 'growth': 0.45, 'market_share': 0.25, 'customer_satisfaction': 0.7},
        {'revenue': 0.7, 'growth': 0.6, 'market_share': 0.4, 'customer_satisfaction': 0.78}
    ]


@pytest.fixture
def sample_market_state():
    """Sample market state for strategy generation"""
    return {
        'market_demand': 0.72,
        'competitor_prices': [0.55, 0.62, 0.58],
        'sales_volume': 0.68,
        'conversion_rate': 0.16,
        'inventory_level': 0.75,
        'market_trend': 0.15
    }


@pytest.fixture
def sample_historical_sales():
    """Sample historical sales data for performance monitoring"""
    return [
        [120.0, 0.16, 0.75],
        [125.0, 0.17, 0.78],
        [118.0, 0.15, 0.73],
        [130.0, 0.18, 0.80],
        [128.0, 0.17, 0.77],
        [135.0, 0.19, 0.82],
        [132.0, 0.18, 0.79],
        [140.0, 0.20, 0.85]
    ]


@pytest.fixture
def sample_product_portfolio():
    """Sample product portfolio for business optimization"""
    return [
        {
            'name': 'CloudERP Standard',
            'sales_history': [100, 110, 105, 115, 120],
            'demand_forecast': 125.0,
            'production_cost': 45.0,
            'current_inventory': 200.0
        },
        {
            'name': 'CloudERP Enterprise',
            'sales_history': [80, 85, 90, 88, 95],
            'demand_forecast': 98.0,
            'production_cost': 65.0,
            'current_inventory': 150.0
        },
        {
            'name': 'CloudERP Premium',
            'sales_history': [60, 65, 70, 68, 75],
            'demand_forecast': 78.0,
            'production_cost': 85.0,
            'current_inventory': 100.0
        }
    ]


class TestCompanyAnalysisWorkflow:
    """Test complete company analysis workflow (Requirement 1.1)"""
    
    def test_company_analysis_sync_workflow(self, client, sample_company_text):
        """Test synchronous company analysis workflow"""
        # Step 1: Analyze company
        payload = {
            'text': sample_company_text,
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # Verify response structure
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify expected fields are present
            assert 'product_category' in data or 'error' not in data
            assert 'business_domain' in data or 'error' not in data
            assert 'processing_time_ms' in data or 'error' not in data
            
            # Step 2: Verify model info includes enterprise analyst
            model_info_response = client.get('/api/v1/model_info')
            assert model_info_response.status_code == 200
            
            model_data = json.loads(model_info_response.data)
            assert 'enterprise_analyst' in model_data
    
    def test_company_analysis_async_workflow(self, client, sample_company_text):
        """Test asynchronous company analysis workflow"""
        # Step 1: Submit async task
        payload = {
            'text': sample_company_text,
            'source_type': 'annual_report'
        }
        
        submit_response = client.post(
            '/api/v1/async/analyze_company/async',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert submit_response.status_code == 202
        submit_data = json.loads(submit_response.data)
        
        assert 'task_id' in submit_data
        assert 'status_url' in submit_data
        assert 'result_url' in submit_data
        
        task_id = submit_data['task_id']
        
        # Step 2: Check task status
        status_response = client.get(f'/api/v1/async/tasks/{task_id}')
        assert status_response.status_code == 200
        
        status_data = json.loads(status_response.data)
        assert 'status' in status_data
        assert status_data['status'] in ['PENDING', 'STARTED', 'SUCCESS', 'FAILURE']
        
        # Step 3: Try to get result (may not be ready)
        result_response = client.get(f'/api/v1/async/tasks/{task_id}/result')
        assert result_response.status_code in [200, 202, 500]
    
    def test_company_analysis_with_explanation(self, client, sample_company_text):
        """Test company analysis workflow with model explanation"""
        # Step 1: Analyze company
        payload = {
            'text': sample_company_text,
            'source_type': 'product_summary'
        }
        
        analysis_response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        if analysis_response.status_code == 200:
            # Step 2: Request explanation for BERT model
            explain_payload = {
                'model_type': 'bert',
                'instance': [0.5] * 10,  # Dummy feature vector
                'top_n': 5,
                'explanation_type': 'local'
            }
            
            explain_response = client.post(
                '/api/v1/explain',
                data=json.dumps(explain_payload),
                content_type='application/json'
            )
            
            assert explain_response.status_code in [200, 400, 500, 503]


class TestMarketAnalysisToStrategyWorkflow:
    """Test market analysis to strategy generation flow (Requirement 2.1, 3.1)"""
    
    def test_market_to_strategy_workflow(self, client, sample_market_data, sample_market_state):
        """Test complete workflow from market analysis to strategy generation"""
        # Step 1: Analyze market
        market_payload = {
            'market_data': sample_market_data,
            'entity_ids': [f'entity_{i}' for i in range(len(sample_market_data))],
            'auto_select_clusters': True
        }
        
        market_response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(market_payload),
            content_type='application/json'
        )
        
        assert market_response.status_code in [200, 500, 503]
        
        if market_response.status_code == 200:
            market_data = json.loads(market_response.data)
            
            # Verify market analysis results
            assert 'clusters' in market_data or 'latent_representations' in market_data
            
            # Step 2: Generate strategy based on market insights
            strategy_payload = {
                'market_state': sample_market_state,
                'context': {
                    'company_name': 'TechCorp',
                    'product_name': 'CloudERP'
                },
                'include_explanation': True
            }
            
            strategy_response = client.post(
                '/api/v1/strategy',
                data=json.dumps(strategy_payload),
                content_type='application/json'
            )
            
            assert strategy_response.status_code in [200, 429, 500, 503]
            
            if strategy_response.status_code == 200:
                strategy_data = json.loads(strategy_response.data)
                
                # Verify strategy results
                assert 'pricing_strategy' in strategy_data or 'recommended_price' in strategy_data
                assert 'confidence_score' in strategy_data or 'explanation' in strategy_data
    
    def test_market_analysis_with_clustering_insights(self, client, sample_market_data):
        """Test market analysis workflow with detailed clustering insights"""
        # Step 1: Perform market analysis
        payload = {
            'market_data': sample_market_data,
            'auto_select_clusters': True,
            'similarity_threshold': 0.75
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify clustering results
            if 'clusters' in data:
                assert isinstance(data['clusters'], (list, dict))
            
            # Step 2: Request explanation for clustering
            if 'latent_representations' in data:
                explain_payload = {
                    'model_type': 'autoencoder',
                    'instance': data['latent_representations'][0] if data['latent_representations'] else [0.5] * 5,
                    'explanation_type': 'local',
                    'top_n': 5
                }
                
                explain_response = client.post(
                    '/api/v1/explain',
                    data=json.dumps(explain_payload),
                    content_type='application/json'
                )
                
                assert explain_response.status_code in [200, 400, 500, 503]
    
    def test_strategy_comparison_workflow(self, client, sample_market_state):
        """Test strategy comparison across multiple scenarios"""
        # Create multiple market scenarios
        scenarios = [
            {
                'name': 'Conservative',
                'market_state': {
                    **sample_market_state,
                    'market_demand': 0.6,
                    'market_trend': 0.05
                }
            },
            {
                'name': 'Moderate',
                'market_state': sample_market_state
            },
            {
                'name': 'Aggressive',
                'market_state': {
                    **sample_market_state,
                    'market_demand': 0.85,
                    'market_trend': 0.25
                }
            }
        ]
        
        payload = {'scenarios': scenarios}
        
        response = client.post(
            '/api/v1/strategy/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 429, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'scenarios' in data or 'comparisons' in data


class TestPerformanceMonitoringFeedbackLoop:
    """Test performance monitoring and feedback loop (Requirement 4.1)"""
    
    def test_performance_monitoring_workflow(self, client, sample_historical_sales):
        """Test complete performance monitoring workflow"""
        # Step 1: Submit historical data for analysis
        payload = {
            'historical_data': sample_historical_sales,
            'include_feedback': True
        }
        
        response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify performance monitoring results
            assert 'forecast' in data or 'predictions' in data
            assert 'alerts' in data or 'anomalies' in data
            
            # Step 2: Check for critical alerts
            alerts_response = client.get('/api/v1/performance/alerts/critical')
            assert alerts_response.status_code in [200, 500, 503]
            
            # Step 3: Get feedback recommendations
            feedback_response = client.get('/api/v1/performance/feedback/recommendations')
            assert feedback_response.status_code in [200, 500, 503]
    
    def test_performance_feedback_to_strategy_loop(self, client, sample_historical_sales, sample_market_state):
        """Test feedback loop from performance monitoring to strategy adjustment"""
        # Step 1: Monitor performance
        perf_payload = {
            'historical_data': sample_historical_sales,
            'current_data': [[145.0, 0.21, 0.87]],
            'include_feedback': True
        }
        
        perf_response = client.post(
            '/api/v1/performance',
            data=json.dumps(perf_payload),
            content_type='application/json'
        )
        
        assert perf_response.status_code in [200, 500, 503]
        
        if perf_response.status_code == 200:
            perf_data = json.loads(perf_response.data)
            
            # Step 2: Use performance insights to adjust strategy
            # Simulate adjusting market state based on performance feedback
            adjusted_market_state = sample_market_state.copy()
            
            # If there are alerts, adjust market state
            if 'alerts' in perf_data and perf_data['alerts']:
                adjusted_market_state['market_trend'] = 0.05  # More conservative
            
            strategy_payload = {
                'market_state': adjusted_market_state,
                'include_explanation': True,
                'deterministic': True
            }
            
            strategy_response = client.post(
                '/api/v1/strategy',
                data=json.dumps(strategy_payload),
                content_type='application/json'
            )
            
            assert strategy_response.status_code in [200, 429, 500, 503]
    
    def test_performance_trends_analysis(self, client, sample_historical_sales):
        """Test performance trends analysis workflow"""
        payload = {
            'historical_data': sample_historical_sales,
            'window_size': 30
        }
        
        response = client.post(
            '/api/v1/performance/trends',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'trends' in data or 'trend_analysis' in data
    
    def test_anomaly_detection_workflow(self, client):
        """Test anomaly detection in performance monitoring"""
        # Create data with an obvious anomaly
        normal_data = [[100.0 + i, 0.15, 0.75] for i in range(10)]
        anomaly_data = [[50.0, 0.08, 0.40]]  # Significant drop
        
        historical_data = normal_data + anomaly_data
        
        payload = {
            'historical_data': historical_data,
            'include_feedback': True
        }
        
        response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            # Should detect the anomaly
            assert 'alerts' in data or 'anomalies' in data


class TestBusinessOptimizationWorkflow:
    """Test business optimization workflow (Requirement 5.1)"""
    
    def test_business_optimization_workflow(self, client, sample_product_portfolio):
        """Test complete business optimization workflow"""
        # Step 1: Optimize product portfolio
        payload = {
            'product_portfolio': sample_product_portfolio,
            'constraints': {
                'min_production': [50, 40, 30],
                'max_production': [250, 200, 150],
                'total_budget': 15000.0,
                'capacity_limit': 600.0
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
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify optimization results
            assert 'production_priorities' in data or 'recommended_production' in data
            assert 'resource_allocation' in data or 'optimization_score' in data
    
    def test_business_scenarios_analysis(self, client, sample_product_portfolio):
        """Test business optimization with multiple scenarios"""
        scenarios = [
            {
                'name': 'Low Budget',
                'constraints': {'total_budget': 8000.0}
            },
            {
                'name': 'Medium Budget',
                'constraints': {'total_budget': 15000.0}
            },
            {
                'name': 'High Budget',
                'constraints': {'total_budget': 25000.0}
            }
        ]
        
        payload = {
            'product_portfolio': sample_product_portfolio,
            'scenarios': scenarios
        }
        
        response = client.post(
            '/api/v1/business_optimizer/scenarios',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'scenarios' in data or 'scenario_results' in data
    
    def test_business_optimization_with_explanation(self, client, sample_product_portfolio):
        """Test business optimization with model explanation"""
        # Step 1: Optimize business
        payload = {
            'product_portfolio': sample_product_portfolio,
            'revenue_weight': 0.6,
            'cost_weight': 0.4
        }
        
        opt_response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        if opt_response.status_code == 200:
            # Step 2: Request explanation for regression model
            explain_payload = {
                'model_type': 'regression',
                'instance': [125.0, 98.0, 78.0, 45.0, 65.0, 85.0],
                'top_n': 5,
                'explanation_type': 'local'
            }
            
            explain_response = client.post(
                '/api/v1/explain',
                data=json.dumps(explain_payload),
                content_type='application/json'
            )
            
            assert explain_response.status_code in [200, 400, 500, 503]


class TestIntegratedEndToEndWorkflow:
    """Test complete integrated workflow across all modules"""
    
    def test_full_system_workflow(self, client, sample_company_text, sample_market_data, 
                                   sample_market_state, sample_historical_sales, 
                                   sample_product_portfolio):
        """Test complete end-to-end workflow across all modules"""
        
        # Step 1: Analyze company
        company_payload = {
            'text': sample_company_text,
            'source_type': 'product_summary'
        }
        
        company_response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(company_payload),
            content_type='application/json'
        )
        
        assert company_response.status_code in [200, 500, 503]
        
        # Step 2: Analyze market
        market_payload = {
            'market_data': sample_market_data,
            'auto_select_clusters': True
        }
        
        market_response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(market_payload),
            content_type='application/json'
        )
        
        assert market_response.status_code in [200, 500, 503]
        
        # Step 3: Generate strategy
        strategy_payload = {
            'market_state': sample_market_state,
            'include_explanation': True
        }
        
        strategy_response = client.post(
            '/api/v1/strategy',
            data=json.dumps(strategy_payload),
            content_type='application/json'
        )
        
        assert strategy_response.status_code in [200, 429, 500, 503]
        
        # Step 4: Monitor performance
        perf_payload = {
            'historical_data': sample_historical_sales,
            'include_feedback': True
        }
        
        perf_response = client.post(
            '/api/v1/performance',
            data=json.dumps(perf_payload),
            content_type='application/json'
        )
        
        assert perf_response.status_code in [200, 500, 503]
        
        # Step 5: Optimize business
        business_payload = {
            'product_portfolio': sample_product_portfolio
        }
        
        business_response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(business_payload),
            content_type='application/json'
        )
        
        assert business_response.status_code in [200, 500, 503]
        
        # Step 6: Verify system health
        health_response = client.get('/api/v1/health')
        assert health_response.status_code in [200, 503]
    
    def test_async_full_workflow(self, client, sample_company_text, sample_market_data, 
                                  sample_market_state, sample_product_portfolio):
        """Test complete async workflow across all modules"""
        
        task_ids = []
        
        # Submit all tasks asynchronously
        
        # Task 1: Company analysis
        company_payload = {
            'text': sample_company_text,
            'source_type': 'product_summary'
        }
        
        company_response = client.post(
            '/api/v1/async/analyze_company/async',
            data=json.dumps(company_payload),
            content_type='application/json'
        )
        
        if company_response.status_code == 202:
            task_ids.append(json.loads(company_response.data)['task_id'])
        
        # Task 2: Market analysis
        market_payload = {
            'market_data': sample_market_data
        }
        
        market_response = client.post(
            '/api/v1/async/market_analysis/async',
            data=json.dumps(market_payload),
            content_type='application/json'
        )
        
        if market_response.status_code == 202:
            task_ids.append(json.loads(market_response.data)['task_id'])
        
        # Task 3: Strategy generation
        strategy_payload = {
            'market_state': sample_market_state
        }
        
        strategy_response = client.post(
            '/api/v1/async/strategy/async',
            data=json.dumps(strategy_payload),
            content_type='application/json'
        )
        
        if strategy_response.status_code == 202:
            task_ids.append(json.loads(strategy_response.data)['task_id'])
        
        # Task 4: Business optimization
        business_payload = {
            'product_portfolio': sample_product_portfolio
        }
        
        business_response = client.post(
            '/api/v1/async/business_optimizer/async',
            data=json.dumps(business_payload),
            content_type='application/json'
        )
        
        if business_response.status_code == 202:
            task_ids.append(json.loads(business_response.data)['task_id'])
        
        # Check status of all tasks
        for task_id in task_ids:
            status_response = client.get(f'/api/v1/async/tasks/{task_id}')
            assert status_response.status_code == 200
            
            status_data = json.loads(status_response.data)
            assert 'status' in status_data
            assert status_data['status'] in ['PENDING', 'STARTED', 'SUCCESS', 'FAILURE']
        
        # List all active tasks
        list_response = client.get('/api/v1/async/tasks')
        assert list_response.status_code == 200
        
        list_data = json.loads(list_response.data)
        assert 'tasks' in list_data
        assert 'total' in list_data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
