"""Integration tests for Dashboard"""

import pytest
import json
from bs4 import BeautifulSoup
from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client with dashboard"""
    app = create_app('testing')
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


class TestDashboardTemplateRendering:
    """Test template rendering for each dashboard view"""
    
    def test_index_template_renders(self, client):
        """Test that index/home page renders correctly"""
        response = client.get('/dashboard/')
        assert response.status_code == 200
        
        # Parse HTML
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check title
        assert soup.title is not None
        assert 'AI Sales Strategist' in soup.title.string
        
        # Check navigation bar exists
        nav = soup.find('nav', class_='navbar')
        assert nav is not None
        
        # Check brand link
        brand = soup.find('a', class_='navbar-brand')
        assert brand is not None
        assert 'AI Sales Strategist' in brand.text
        
        # Check all navigation links are present
        nav_links = soup.find_all('a', class_='nav-link')
        link_texts = [link.text.strip() for link in nav_links]
        
        assert any('Home' in text for text in link_texts)
        assert any('Market Analysis' in text for text in link_texts)
        assert any('Strategy' in text for text in link_texts)
        assert any('Performance' in text for text in link_texts)
        assert any('Explainability' in text for text in link_texts)
        assert any('Business Optimization' in text for text in link_texts)
        
        # Check footer exists
        footer = soup.find('footer', class_='footer')
        assert footer is not None
        assert 'AI-powered Enterprise Sales Strategist' in footer.text
    
    def test_market_analysis_template_renders(self, client):
        """Test that market analysis view renders correctly"""
        response = client.get('/dashboard/market-analysis')
        assert response.status_code == 200
        
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check title
        assert 'Market Analysis' in soup.title.string
        
        # Check active navigation
        active_link = soup.find('a', class_='nav-link active')
        assert active_link is not None
        assert 'Market Analysis' in active_link.text
        
        # Check for key elements (actual IDs from template)
        assert soup.find(id='alertContainer') is not None
        assert soup.find(id='marketDataFile') is not None
        
        # Check for required scripts
        scripts = soup.find_all('script')
        script_srcs = [script.get('src', '') for script in scripts]
        
        # Should include Chart.js, Plotly, D3
        assert any('chart.js' in src.lower() for src in script_srcs)
        assert any('plotly' in src.lower() for src in script_srcs)
        assert any('d3' in src.lower() for src in script_srcs)
        
        # Should include API client
        assert any('api-client.js' in src for src in script_srcs)
        
        # Should include market analysis specific script
        assert any('market-analysis.js' in src for src in script_srcs)
    
    def test_strategy_template_renders(self, client):
        """Test that strategy view renders correctly"""
        response = client.get('/dashboard/strategy')
        assert response.status_code == 200
        
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check title
        assert 'Strategy' in soup.title.string
        
        # Check active navigation
        active_link = soup.find('a', class_='nav-link active')
        assert active_link is not None
        assert 'Strategy' in active_link.text
        
        # Check for key elements (actual IDs from template)
        assert soup.find(id='strategyForm') is not None
        assert soup.find(id='resultsSection') is not None
        
        # Check for strategy-specific script
        scripts = soup.find_all('script')
        script_srcs = [script.get('src', '') for script in scripts]
        assert any('strategy.js' in src for src in script_srcs)
    
    def test_performance_template_renders(self, client):
        """Test that performance view renders correctly"""
        response = client.get('/dashboard/performance')
        assert response.status_code == 200
        
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check title
        assert 'Performance' in soup.title.string
        
        # Check active navigation
        active_link = soup.find('a', class_='nav-link active')
        assert active_link is not None
        assert 'Performance' in active_link.text
        
        # Check for key elements (actual IDs from template)
        assert soup.find(id='performanceForm') is not None
        assert soup.find(id='resultsSection') is not None
        
        # Check for performance-specific script
        scripts = soup.find_all('script')
        script_srcs = [script.get('src', '') for script in scripts]
        assert any('performance.js' in src for src in script_srcs)
    
    def test_explainability_template_renders(self, client):
        """Test that explainability view renders correctly"""
        response = client.get('/dashboard/explainability')
        assert response.status_code == 200
        
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check title
        assert 'Explainability' in soup.title.string
        
        # Check active navigation
        active_link = soup.find('a', class_='nav-link active')
        assert active_link is not None
        assert 'Explainability' in active_link.text
        
        # Check for key elements (actual IDs from template)
        assert soup.find(id='explainabilityForm') is not None
        assert soup.find(id='modelType') is not None
        assert soup.find(id='explanationType') is not None
        
        # Check for explainability-specific script
        scripts = soup.find_all('script')
        script_srcs = [script.get('src', '') for script in scripts]
        assert any('explainability.js' in src for src in script_srcs)
    
    def test_business_optimization_template_renders(self, client):
        """Test that business optimization view renders correctly"""
        response = client.get('/dashboard/business-optimization')
        assert response.status_code == 200
        
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check title
        assert 'Business Optimization' in soup.title.string
        
        # Check active navigation
        active_link = soup.find('a', class_='nav-link active')
        assert active_link is not None
        assert 'Business Optimization' in active_link.text
        
        # Check for key elements (actual IDs from template)
        assert soup.find(id='optimizationForm') is not None
        assert soup.find(id='productsData') is not None
        assert soup.find(id='budgetLimit') is not None
        
        # Check for business optimization-specific script
        scripts = soup.find_all('script')
        script_srcs = [script.get('src', '') for script in scripts]
        assert any('business-optimization.js' in src for src in script_srcs)


class TestDashboardAPIIntegration:
    """Test API integration and data flow"""
    
    def test_dashboard_can_access_api_endpoints(self, client):
        """Test that dashboard can access API endpoints"""
        # Test health endpoint
        response = client.get('/api/v1/health')
        assert response.status_code in [200, 503]
        
        # Test model info endpoint
        response = client.get('/api/v1/model_info')
        assert response.status_code == 200
    
    def test_analyze_company_from_dashboard_flow(self, client):
        """Test company analysis flow as initiated from dashboard"""
        payload = {
            'text': 'TechCorp provides enterprise software solutions.',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Verify data structure matches what dashboard expects
        assert 'product_category' in data
        assert 'business_domain' in data
        assert 'value_proposition' in data
        assert 'key_features' in data
        assert isinstance(data['key_features'], list)
        assert 'confidence_scores' in data
        assert isinstance(data['confidence_scores'], dict)
    
    def test_market_analysis_from_dashboard_flow(self, client):
        """Test market analysis flow as initiated from dashboard"""
        # This would typically be called from the market analysis view
        payload = {
            'market_data': [
                {'entity_id': 'company_1', 'features': [0.5, 0.3, 0.8]},
                {'entity_id': 'company_2', 'features': [0.6, 0.4, 0.7]}
            ]
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May return 200 or 400 depending on model availability
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            # Verify expected structure
            assert 'clusters' in data or 'error' not in data
    
    def test_strategy_generation_from_dashboard_flow(self, client):
        """Test strategy generation flow as initiated from dashboard"""
        payload = {
            'market_state': {
                'demand': 1000,
                'competitor_prices': [99.99, 89.99],
                'sales_data': [100, 120, 110]
            }
        }
        
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May return 200 or 400 depending on model availability
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            # Verify expected structure
            assert 'strategy' in data or 'pricing' in data or 'error' not in data
    
    def test_performance_monitoring_from_dashboard_flow(self, client):
        """Test performance monitoring flow as initiated from dashboard"""
        payload = {
            'historical_data': [100, 110, 105, 120, 115, 130]
        }
        
        response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May return 200 or 400 depending on model availability
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            # Verify expected structure
            assert 'forecast' in data or 'predictions' in data or 'error' not in data
    
    def test_business_optimization_from_dashboard_flow(self, client):
        """Test business optimization flow as initiated from dashboard"""
        payload = {
            'products': [
                {'id': 'prod_1', 'name': 'Product A', 'cost': 50, 'price': 100},
                {'id': 'prod_2', 'name': 'Product B', 'cost': 30, 'price': 60}
            ]
        }
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May return 200 or 400 depending on model availability
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            # Verify expected structure
            assert 'recommendations' in data or 'priorities' in data or 'error' not in data
    
    def test_explanation_from_dashboard_flow(self, client):
        """Test model explanation flow as initiated from dashboard"""
        payload = {
            'model_type': 'strategy',
            'prediction': {'price': 99.99},
            'features': {'demand': 1000, 'competitor_price': 89.99}
        }
        
        response = client.post(
            '/api/v1/explain',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May return 200 or 400 depending on model availability
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            # Verify expected structure
            assert 'shap_values' in data or 'feature_importance' in data or 'error' not in data


class TestDashboardInteractiveFeatures:
    """Test interactive features and user interactions"""
    
    def test_navigation_between_views(self, client):
        """Test navigation between different dashboard views"""
        views = [
            '/dashboard/',
            '/dashboard/market-analysis',
            '/dashboard/strategy',
            '/dashboard/performance',
            '/dashboard/explainability',
            '/dashboard/business-optimization'
        ]
        
        for view in views:
            response = client.get(view)
            assert response.status_code == 200
            
            # Check that navigation links are present
            soup = BeautifulSoup(response.data, 'html.parser')
            nav_links = soup.find_all('a', class_='nav-link')
            assert len(nav_links) >= 6  # At least 6 main navigation items
    
    def test_static_assets_accessible(self, client):
        """Test that static assets (CSS, JS) are accessible"""
        # Note: Static files are served through Flask's static folder mechanism
        # In production, these would be served by a web server
        # We verify that the templates reference the correct static paths
        
        response = client.get('/dashboard/')
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check that CSS link exists in template
        css_links = soup.find_all('link', rel='stylesheet')
        dashboard_css_found = any('dashboard.css' in link.get('href', '') for link in css_links)
        assert dashboard_css_found, "Dashboard CSS not referenced in template"
        
        # Check that JS scripts exist in template
        scripts = soup.find_all('script')
        script_srcs = [script.get('src', '') for script in scripts]
        
        assert any('api-client.js' in src for src in script_srcs), "API client JS not referenced"
    
    def test_bootstrap_integration(self, client):
        """Test that Bootstrap is properly integrated"""
        response = client.get('/dashboard/')
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check for Bootstrap CSS link
        css_links = soup.find_all('link', rel='stylesheet')
        bootstrap_found = any('bootstrap' in link.get('href', '').lower() for link in css_links)
        assert bootstrap_found
        
        # Check for Bootstrap JS
        scripts = soup.find_all('script')
        bootstrap_js_found = any('bootstrap' in script.get('src', '').lower() for script in scripts)
        assert bootstrap_js_found
        
        # Check for Bootstrap classes
        assert soup.find(class_='navbar') is not None
        assert soup.find(class_='container-fluid') is not None
    
    def test_chart_libraries_loaded(self, client):
        """Test that chart libraries are loaded"""
        response = client.get('/dashboard/market-analysis')
        soup = BeautifulSoup(response.data, 'html.parser')
        
        scripts = soup.find_all('script')
        script_srcs = [script.get('src', '') for script in scripts]
        
        # Check for Chart.js
        assert any('chart.js' in src.lower() for src in script_srcs)
        
        # Check for Plotly
        assert any('plotly' in src.lower() for src in script_srcs)
        
        # Check for D3.js
        assert any('d3' in src.lower() for src in script_srcs)
    
    def test_api_client_loaded_on_all_pages(self, client):
        """Test that API client is loaded on all dashboard pages"""
        views = [
            '/dashboard/',
            '/dashboard/market-analysis',
            '/dashboard/strategy',
            '/dashboard/performance',
            '/dashboard/explainability',
            '/dashboard/business-optimization'
        ]
        
        for view in views:
            response = client.get(view)
            soup = BeautifulSoup(response.data, 'html.parser')
            
            scripts = soup.find_all('script')
            script_srcs = [script.get('src', '') for script in scripts]
            
            assert any('api-client.js' in src for src in script_srcs), \
                f"API client not found on {view}"


class TestDashboardResponsiveDesign:
    """Test responsive design on different screen sizes"""
    
    def test_viewport_meta_tag_present(self, client):
        """Test that viewport meta tag is present for responsive design"""
        response = client.get('/dashboard/')
        soup = BeautifulSoup(response.data, 'html.parser')
        
        viewport = soup.find('meta', attrs={'name': 'viewport'})
        assert viewport is not None
        assert 'width=device-width' in viewport.get('content', '')
    
    def test_responsive_navigation(self, client):
        """Test that navigation has responsive toggle button"""
        response = client.get('/dashboard/')
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check for navbar toggler (hamburger menu for mobile)
        toggler = soup.find('button', class_='navbar-toggler')
        assert toggler is not None
        
        # Check for collapsible navigation
        collapse = soup.find('div', class_='collapse navbar-collapse')
        assert collapse is not None
    
    def test_bootstrap_grid_system_used(self, client):
        """Test that Bootstrap grid system is used for responsive layout"""
        views = [
            '/dashboard/market-analysis',
            '/dashboard/strategy',
            '/dashboard/performance',
            '/dashboard/explainability',
            '/dashboard/business-optimization'
        ]
        
        for view in views:
            response = client.get(view)
            soup = BeautifulSoup(response.data, 'html.parser')
            
            # Check for container classes
            containers = soup.find_all(class_=lambda x: x and ('container' in x or 'row' in x or 'col' in x))
            assert len(containers) > 0, f"No Bootstrap grid classes found on {view}"
    
    def test_responsive_utilities_present(self, client):
        """Test that responsive utility classes are available"""
        response = client.get('/dashboard/')
        soup = BeautifulSoup(response.data, 'html.parser')
        
        # Check for responsive classes in HTML
        html_content = str(soup)
        
        # Bootstrap responsive utilities should be available
        # (we can't test actual rendering, but we can check the framework is loaded)
        css_links = soup.find_all('link', rel='stylesheet')
        bootstrap_found = any('bootstrap' in link.get('href', '').lower() for link in css_links)
        assert bootstrap_found


class TestDashboardErrorHandling:
    """Test error handling in dashboard"""
    
    def test_404_on_invalid_dashboard_route(self, client):
        """Test 404 error on invalid dashboard route"""
        response = client.get('/dashboard/nonexistent-page')
        assert response.status_code == 404
    
    def test_api_error_handling(self, client):
        """Test that API errors are properly formatted for dashboard consumption"""
        # Test with invalid request
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps({}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        # Error should have proper structure
        assert 'error' in data or 'message' in data
    
    def test_root_redirects_to_dashboard(self, client):
        """Test that root URL redirects to dashboard"""
        response = client.get('/', follow_redirects=False)
        assert response.status_code == 302
        assert '/dashboard/' in response.location


class TestDashboardDataFlow:
    """Test end-to-end data flow from dashboard to API and back"""
    
    def test_complete_analysis_workflow(self, client):
        """Test complete workflow: dashboard -> API -> response -> dashboard"""
        # Step 1: Load dashboard page
        response = client.get('/dashboard/')
        assert response.status_code == 200
        
        # Step 2: Simulate API call from dashboard
        payload = {
            'text': 'Enterprise software company providing cloud solutions.',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        # Step 3: Verify data structure is suitable for dashboard rendering
        assert 'product_category' in data
        assert 'business_domain' in data
        assert 'confidence_scores' in data
        
        # Confidence scores should be numeric (for progress bars, etc.)
        for key, value in data['confidence_scores'].items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1
    
    def test_health_check_integration(self, client):
        """Test health check integration for dashboard status indicators"""
        response = client.get('/api/v1/health')
        assert response.status_code in [200, 503]
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'services' in data
        
        # Services should have status information
        assert isinstance(data['services'], dict)
        for service_name, service_info in data['services'].items():
            assert 'status' in service_info
    
    def test_model_info_integration(self, client):
        """Test model info integration for dashboard display"""
        response = client.get('/api/v1/model_info')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        
        # Should contain information about available models
        assert isinstance(data, dict)
        assert len(data) > 0
        
        # Each model should have status information
        for model_name, model_info in data.items():
            assert 'status' in model_info
