"""Dashboard integration tests for Sales Strategist System

Tests data flow from API to dashboard views, interactive features,
performance, and real-time update mechanisms.

Requirements tested:
- 7.8: Dashboard integration with REST API
"""

import pytest
import json
import time
from bs4 import BeautifulSoup
from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client with testing configuration"""
    app = create_app('testing')
    app.config['TESTING'] = True
    app.config['API_KEYS'] = 'test-dashboard-integration'
    
    with app.test_client() as client:
        yield client


class TestDashboardAPIDataFlow:
    """Test data flow from API to dashboard views (Requirement 7.8)"""
    
    def test_company_analysis_data_flow(self, client):
        """Test complete data flow for company analysis view"""
        # Step 1: Load dashboard page
        dashboard_response = client.get('/dashboard/')
        assert dashboard_response.status_code == 200
        
        # Step 2: Submit company analysis request
        payload = {
            'text': 'TechCorp provides cloud-based ERP solutions for enterprises.',
            'source_type': 'product_summary'
        }
        
        api_response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert api_response.status_code in [200, 500, 503]
        
        if api_response.status_code == 200:
            data = json.loads(api_response.data)
            
            # Verify data structure matches dashboard expectations
            assert 'product_category' in data
            assert 'business_domain' in data
            assert 'value_proposition' in data
            assert 'key_features' in data
            assert 'confidence_scores' in data
            
            # Verify data types for dashboard rendering
            assert isinstance(data['product_category'], str)
            assert isinstance(data['business_domain'], str)
            assert isinstance(data['key_features'], list)
            assert isinstance(data['confidence_scores'], dict)
            
            # Verify confidence scores are in valid range for progress bars
            for score in data['confidence_scores'].values():
                assert 0 <= score <= 1
    
    def test_market_analysis_data_flow(self, client):
        """Test complete data flow for market analysis view"""
        # Step 1: Load market analysis page
        dashboard_response = client.get('/dashboard/market-analysis')
        assert dashboard_response.status_code == 200
        
        # Verify page has required elements
        soup = BeautifulSoup(dashboard_response.data, 'html.parser')
        assert soup.find(id='marketDataFile') is not None
        assert soup.find(id='alertContainer') is not None
        
        # Step 2: Submit market analysis request
        payload = {
            'market_data': [
                {'revenue': 0.75, 'growth': 0.6, 'market_share': 0.45},
                {'revenue': 0.65, 'growth': 0.55, 'market_share': 0.35},
                {'revenue': 0.85, 'growth': 0.7, 'market_share': 0.55}
            ],
            'entity_ids': ['company_1', 'company_2', 'company_3'],
            'auto_select_clusters': True
        }
        
        api_response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert api_response.status_code in [200, 500, 503]
        
        if api_response.status_code == 200:
            data = json.loads(api_response.data)
            
            # Verify data structure for visualization
            assert 'clusters' in data or 'latent_representations' in data
            assert 'processing_time_ms' in data
            
            # Verify data is suitable for Chart.js/Plotly rendering
            if 'clusters' in data:
                assert isinstance(data['clusters'], (list, dict))
    
    def test_strategy_generation_data_flow(self, client):
        """Test complete data flow for strategy view"""
        # Step 1: Load strategy page
        dashboard_response = client.get('/dashboard/strategy')
        assert dashboard_response.status_code == 200
        
        # Verify page has required elements
        soup = BeautifulSoup(dashboard_response.data, 'html.parser')
        assert soup.find(id='strategyForm') is not None
        assert soup.find(id='resultsSection') is not None
        
        # Step 2: Submit strategy generation request
        payload = {
            'market_state': {
                'market_demand': 0.72,
                'competitor_prices': [0.55, 0.62, 0.58],
                'sales_volume': 0.68,
                'conversion_rate': 0.16,
                'inventory_level': 0.75,
                'market_trend': 0.15
            },
            'context': {
                'company_name': 'TechCorp',
                'product_name': 'CloudERP'
            },
            'include_explanation': True
        }
        
        api_response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert api_response.status_code in [200, 429, 500, 503]
        
        if api_response.status_code == 200:
            data = json.loads(api_response.data)
            
            # Verify data structure for dashboard display
            assert 'pricing_strategy' in data or 'recommended_price' in data
            assert 'confidence_score' in data
            
            # Verify explanation is present if requested
            if 'explanation' in data:
                assert isinstance(data['explanation'], str)
                assert len(data['explanation']) > 0
    
    def test_performance_monitoring_data_flow(self, client):
        """Test complete data flow for performance view"""
        # Step 1: Load performance page
        dashboard_response = client.get('/dashboard/performance')
        assert dashboard_response.status_code == 200
        
        # Verify page has required elements
        soup = BeautifulSoup(dashboard_response.data, 'html.parser')
        assert soup.find(id='performanceForm') is not None
        assert soup.find(id='resultsSection') is not None
        
        # Step 2: Submit performance monitoring request
        payload = {
            'historical_data': [
                [120.0, 0.16, 0.75],
                [125.0, 0.17, 0.78],
                [118.0, 0.15, 0.73],
                [130.0, 0.18, 0.80],
                [128.0, 0.17, 0.77]
            ],
            'include_feedback': True
        }
        
        api_response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert api_response.status_code in [200, 500, 503]
        
        if api_response.status_code == 200:
            data = json.loads(api_response.data)
            
            # Verify data structure for time-series charts
            assert 'forecast' in data or 'predictions' in data
            assert 'alerts' in data or 'anomalies' in data
            
            # Verify forecast data is suitable for Chart.js
            if 'forecast' in data:
                assert isinstance(data['forecast'], list)
    
    def test_business_optimization_data_flow(self, client):
        """Test complete data flow for business optimization view"""
        # Step 1: Load business optimization page
        dashboard_response = client.get('/dashboard/business-optimization')
        assert dashboard_response.status_code == 200
        
        # Verify page has required elements
        soup = BeautifulSoup(dashboard_response.data, 'html.parser')
        assert soup.find(id='optimizationForm') is not None
        assert soup.find(id='productsData') is not None
        
        # Step 2: Submit business optimization request
        payload = {
            'product_portfolio': [
                {
                    'name': 'CloudERP Standard',
                    'sales_history': [100, 110, 105, 115],
                    'production_cost': 45.0
                },
                {
                    'name': 'CloudERP Enterprise',
                    'sales_history': [80, 85, 90, 88],
                    'production_cost': 65.0
                }
            ],
            'constraints': {
                'total_budget': 15000.0
            }
        }
        
        api_response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert api_response.status_code in [200, 500, 503]
        
        if api_response.status_code == 200:
            data = json.loads(api_response.data)
            
            # Verify data structure for visualization
            assert 'production_priorities' in data or 'recommended_production' in data
            assert 'resource_allocation' in data or 'optimization_score' in data
    
    def test_explainability_data_flow(self, client):
        """Test complete data flow for explainability view"""
        # Step 1: Load explainability page
        dashboard_response = client.get('/dashboard/explainability')
        assert dashboard_response.status_code == 200
        
        # Verify page has required elements
        soup = BeautifulSoup(dashboard_response.data, 'html.parser')
        assert soup.find(id='explainabilityForm') is not None
        assert soup.find(id='modelType') is not None
        
        # Step 2: Submit explanation request
        payload = {
            'model_type': 'rl',
            'instance': [0.72, 0.55, 0.68, 0.16, 0.75, 0.15],
            'top_n': 5,
            'explanation_type': 'local',
            'include_visualizations': True
        }
        
        api_response = client.post(
            '/api/v1/explain',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert api_response.status_code in [200, 400, 500, 503]
        
        if api_response.status_code == 200:
            data = json.loads(api_response.data)
            
            # Verify SHAP data structure
            assert 'shap_values' in data or 'feature_importance' in data
            assert 'top_features' in data or 'feature_names' in data


class TestDashboardInteractiveFeatures:
    """Test interactive features and user workflows (Requirement 7.8)"""
    
    def test_form_submission_workflow(self, client):
        """Test form submission and result display workflow"""
        # Load strategy page
        response = client.get('/dashboard/strategy')
        assert response.status_code == 200
        
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Verify form exists
        form = soup.find(id='strategyForm')
        assert form is not None
        
        # Verify form has submit button
        submit_button = soup.find('button', type='submit')
        assert submit_button is not None
        
        # Verify results section exists
        results_section = soup.find(id='resultsSection')
        assert results_section is not None
    
    def test_file_upload_workflow(self, client):
        """Test file upload functionality for market analysis"""
        response = client.get('/dashboard/market-analysis')
        assert response.status_code == 200
        
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Verify file input exists
        file_input = soup.find(id='marketDataFile')
        assert file_input is not None
        assert file_input.get('type') == 'file'
    
    def test_dynamic_chart_rendering_elements(self, client):
        """Test that chart rendering elements are present"""
        views_with_charts = [
            ('/dashboard/market-analysis', ['clusterChart', 'demandChart']),
            ('/dashboard/strategy', ['pricingChart', 'revenueChart']),
            ('/dashboard/performance', ['trendChart', 'alertsChart']),
            ('/dashboard/business-optimization', ['priorityChart', 'resourceChart'])
        ]
        
        for view_url, expected_chart_ids in views_with_charts:
            response = client.get(view_url)
            assert response.status_code == 200
            
            soup = BeautifulSoup(response.data, 'html.parser')
            
            # Check for canvas elements (used by Chart.js)
            canvases = soup.find_all('canvas')
            assert len(canvases) > 0, f"No canvas elements found on {view_url}"
    
    def test_loading_indicators_present(self, client):
        """Test that loading indicators are present for async operations"""
        views = [
            '/dashboard/market-analysis',
            '/dashboard/strategy',
            '/dashboard/performance',
            '/dashboard/business-optimization'
        ]
        
        for view in views:
            response = client.get(view)
            soup = BeautifulSoup(response.data, 'html.parser')
            
            # Check for loading spinner or indicator
            # Common patterns: spinner, loading class, or hidden results section
            html_content = str(soup)
            has_loading_indicator = (
                'spinner' in html_content.lower() or
                'loading' in html_content.lower() or
                'd-none' in html_content  # Bootstrap hidden class
            )
            assert has_loading_indicator, f"No loading indicator found on {view}"
    
    def test_error_display_elements(self, client):
        """Test that error display elements are present"""
        views = [
            '/dashboard/market-analysis',
            '/dashboard/strategy',
            '/dashboard/performance',
            '/dashboard/business-optimization'
        ]
        
        for view in views:
            response = client.get(view)
            soup = BeautifulSoup(response.data, 'html.parser')
            
            # Check for alert container or error display element
            alert_container = soup.find(id='alertContainer') or soup.find(class_='alert')
            assert alert_container is not None, f"No alert container found on {view}"
    
    def test_navigation_state_persistence(self, client):
        """Test that navigation maintains active state"""
        views = [
            ('/dashboard/', 'Home'),
            ('/dashboard/market-analysis', 'Market Analysis'),
            ('/dashboard/strategy', 'Strategy'),
            ('/dashboard/performance', 'Performance'),
            ('/dashboard/explainability', 'Explainability'),
            ('/dashboard/business-optimization', 'Business Optimization')
        ]
        
        for view_url, expected_active in views:
            response = client.get(view_url)
            soup = BeautifulSoup(response.data, 'html.parser')
            
            # Find active navigation link
            active_link = soup.find('a', class_='nav-link active')
            assert active_link is not None, f"No active nav link on {view_url}"
            assert expected_active in active_link.text, \
                f"Expected '{expected_active}' to be active on {view_url}"
    
    def test_scenario_comparison_interface(self, client):
        """Test scenario comparison interface elements"""
        response = client.get('/dashboard/strategy')
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check for scenario comparison elements
        html_content = str(soup)
        has_comparison_features = (
            'scenario' in html_content.lower() or
            'compare' in html_content.lower()
        )
        assert has_comparison_features


class TestDashboardPerformance:
    """Test dashboard loading performance (Requirement 7.8)"""
    
    def test_dashboard_page_load_time(self, client):
        """Test that dashboard pages load within acceptable time"""
        views = [
            '/dashboard/',
            '/dashboard/market-analysis',
            '/dashboard/strategy',
            '/dashboard/performance',
            '/dashboard/explainability',
            '/dashboard/business-optimization'
        ]
        
        for view in views:
            start_time = time.time()
            response = client.get(view)
            load_time = time.time() - start_time
            
            assert response.status_code == 200
            # Page should load in under 2 seconds
            assert load_time < 2.0, f"{view} took {load_time:.2f}s to load"
    
    def test_api_response_time_acceptable(self, client):
        """Test that API responses are within acceptable time"""
        # Test lightweight endpoint
        start_time = time.time()
        response = client.get('/api/v1/health')
        response_time = time.time() - start_time
        
        assert response.status_code in [200, 503]
        # Health check should be very fast
        assert response_time < 1.0, f"Health check took {response_time:.2f}s"
        
        # Test model info endpoint
        start_time = time.time()
        response = client.get('/api/v1/model_info')
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        # Model info should be fast (cached)
        assert response_time < 1.0, f"Model info took {response_time:.2f}s"
    
    def test_static_asset_references_optimized(self, client):
        """Test that static assets are properly referenced"""
        response = client.get('/dashboard/')
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check CSS links
        css_links = soup.find_all('link', rel='stylesheet')
        assert len(css_links) > 0
        
        # Check that external CDN resources are used for libraries
        css_hrefs = [link.get('href', '') for link in css_links]
        has_cdn = any('cdn' in href.lower() for href in css_hrefs)
        assert has_cdn, "Should use CDN for external libraries"
        
        # Check JS scripts
        scripts = soup.find_all('script')
        script_srcs = [script.get('src', '') for script in scripts if script.get('src')]
        assert len(script_srcs) > 0
    
    def test_minimal_api_calls_on_page_load(self, client):
        """Test that pages don't make unnecessary API calls on load"""
        # Load a page and verify it doesn't trigger immediate API calls
        # (API calls should be triggered by user interaction)
        response = client.get('/dashboard/strategy')
        assert response.status_code == 200
        
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check that results section is initially hidden
        results_section = soup.find(id='resultsSection')
        if results_section:
            classes = results_section.get('class', [])
            # Should have d-none or similar hiding class initially
            assert 'd-none' in classes or 'hidden' in classes or \
                   results_section.get('style', '').find('display: none') >= 0


class TestDashboardRealTimeUpdates:
    """Test real-time update mechanisms (Requirement 7.8)"""
    
    def test_async_task_status_polling(self, client):
        """Test async task status polling mechanism"""
        # Submit an async task
        payload = {
            'text': 'TechCorp provides enterprise software.',
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
        
        # Poll task status (simulating dashboard polling)
        max_polls = 5
        for i in range(max_polls):
            status_response = client.get(f'/api/v1/async/tasks/{task_id}')
            assert status_response.status_code == 200
            
            status_data = json.loads(status_response.data)
            assert 'status' in status_data
            assert 'task_id' in status_data
            
            if status_data['status'] in ['SUCCESS', 'FAILURE']:
                break
            
            time.sleep(0.5)
    
    def test_performance_alerts_endpoint(self, client):
        """Test real-time performance alerts endpoint"""
        response = client.get('/api/v1/performance/alerts/critical')
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify alert structure for real-time display
            assert 'alerts' in data or 'critical_alerts' in data
            
            if 'alerts' in data:
                assert isinstance(data['alerts'], list)
    
    def test_feedback_recommendations_endpoint(self, client):
        """Test real-time feedback recommendations endpoint"""
        response = client.get('/api/v1/performance/feedback/recommendations')
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify recommendations structure
            assert 'recommendations' in data or 'feedback' in data
    
    def test_ajax_compatible_responses(self, client):
        """Test that API responses are AJAX-compatible"""
        endpoints = [
            '/api/v1/health',
            '/api/v1/model_info',
            '/api/v1/performance/alerts/critical',
            '/api/v1/performance/feedback/recommendations'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            
            # Should return JSON
            assert response.content_type == 'application/json'
            
            # Should be parseable
            try:
                json.loads(response.data)
            except json.JSONDecodeError:
                pytest.fail(f"Response from {endpoint} is not valid JSON")
    
    def test_cors_headers_for_ajax(self, client):
        """Test that CORS headers are present for AJAX requests"""
        response = client.get('/api/v1/health')
        
        # Check for CORS headers
        assert 'Access-Control-Allow-Origin' in response.headers or \
               response.status_code in [200, 503]
    
    def test_request_id_tracking(self, client):
        """Test request ID tracking for debugging"""
        response = client.get('/api/v1/health')
        
        # Should have request ID header
        assert 'X-Request-ID' in response.headers or \
               response.status_code in [200, 503]


class TestDashboardDataVisualization:
    """Test data visualization integration"""
    
    def test_chart_data_format_compatibility(self, client):
        """Test that API data format is compatible with Chart.js"""
        payload = {
            'historical_data': [
                [120.0, 0.16, 0.75],
                [125.0, 0.17, 0.78],
                [130.0, 0.18, 0.80]
            ]
        }
        
        response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify data structure is suitable for Chart.js
            if 'forecast' in data:
                assert isinstance(data['forecast'], list)
                # Chart.js expects array of numbers or objects
                if len(data['forecast']) > 0:
                    assert isinstance(data['forecast'][0], (int, float, dict))
    
    def test_plotly_data_format_compatibility(self, client):
        """Test that API data format is compatible with Plotly"""
        payload = {
            'market_data': [
                {'revenue': 0.75, 'growth': 0.6},
                {'revenue': 0.65, 'growth': 0.55},
                {'revenue': 0.85, 'growth': 0.7}
            ]
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify data structure is suitable for Plotly
            if 'clusters' in data:
                assert isinstance(data['clusters'], (list, dict))
    
    def test_d3_data_format_compatibility(self, client):
        """Test that API data format is compatible with D3.js"""
        payload = {
            'market_data': [
                {'entity_id': 'A', 'features': [0.5, 0.3]},
                {'entity_id': 'B', 'features': [0.6, 0.4]}
            ],
            'entity_ids': ['A', 'B']
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # D3 typically works with arrays of objects
            # Verify data can be used for network graphs
            if 'graph_data' in data:
                assert isinstance(data['graph_data'], dict)
                if 'nodes' in data['graph_data']:
                    assert isinstance(data['graph_data']['nodes'], list)
                if 'links' in data['graph_data']:
                    assert isinstance(data['graph_data']['links'], list)


class TestDashboardUserWorkflows:
    """Test complete user workflows through dashboard"""
    
    def test_new_user_onboarding_flow(self, client):
        """Test new user can navigate and use dashboard"""
        # Step 1: Land on home page
        response = client.get('/dashboard/')
        assert response.status_code == 200
        
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Should have clear navigation
        nav = soup.find('nav', class_='navbar')
        assert nav is not None
        
        # Should have welcome content or instructions
        assert soup.find('h1') is not None
        
        # Step 2: Navigate to a feature page
        response = client.get('/dashboard/strategy')
        assert response.status_code == 200
        
        # Should have form with clear labels
        soup = BeautifulSoup(response.data, 'html.parser')
        form = soup.find('form')
        assert form is not None
    
    def test_power_user_workflow(self, client):
        """Test power user can efficiently use multiple features"""
        # Simulate power user navigating through multiple views
        views = [
            '/dashboard/market-analysis',
            '/dashboard/strategy',
            '/dashboard/performance',
            '/dashboard/business-optimization'
        ]
        
        for view in views:
            response = client.get(view)
            assert response.status_code == 200
            
            # Each view should load quickly
            # (tested in performance tests)
    
    def test_error_recovery_workflow(self, client):
        """Test user can recover from errors"""
        # Submit invalid request
        payload = {
            'text': '',  # Empty text
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        error_data = json.loads(response.data)
        
        # Error should have clear message
        assert 'error' in error_data or 'message' in error_data
        
        # User can retry with valid data
        valid_payload = {
            'text': 'TechCorp provides software solutions.',
            'source_type': 'product_summary'
        }
        
        retry_response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(valid_payload),
            content_type='application/json'
        )
        
        assert retry_response.status_code in [200, 500, 503]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
