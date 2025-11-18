"""Tests for API endpoints"""

import pytest
import json
from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client"""
    app = create_app('testing')
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


class TestAPIEndpoints:
    """Test cases for API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get('/')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'service' in data
        assert 'version' in data
        assert 'status' in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/api/v1/health')
        assert response.status_code in [200, 503]
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] in ['healthy', 'unhealthy']
        assert 'services' in data
        assert 'enterprise_analyst' in data['services']
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get('/api/v1/model_info')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'enterprise_analyst' in data
        assert 'status' in data['enterprise_analyst']
    
    # Analyze Company Endpoint Tests with Various Input Formats
    
    def test_analyze_company_success(self, client):
        """Test successful company analysis"""
        payload = {
            'text': 'TechCorp provides cloud-based software solutions for enterprises.',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'product_category' in data
        assert 'business_domain' in data
        assert 'value_proposition' in data
        assert 'key_features' in data
        assert 'confidence_scores' in data
        assert 'processing_time_ms' in data
        
        # Verify data types
        assert isinstance(data['product_category'], str)
        assert isinstance(data['business_domain'], str)
        assert isinstance(data['key_features'], list)
        assert isinstance(data['confidence_scores'], dict)
        assert isinstance(data['processing_time_ms'], int)
    
    def test_analyze_company_without_source_type(self, client):
        """Test analysis without optional source_type field"""
        payload = {
            'text': 'TechCorp provides cloud-based software solutions for enterprises.'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'product_category' in data
        assert 'business_domain' in data
    
    def test_analyze_company_annual_report_format(self, client):
        """Test analysis with annual report format"""
        payload = {
            'text': '''
                Annual Report 2023: TechCorp International achieved record revenue
                of $500M, representing 25% year-over-year growth. Our enterprise
                software division continues to be our primary revenue driver, with
                strong adoption in the healthcare and financial services sectors.
            ''',
            'source_type': 'annual_report'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['source_type'] == 'annual_report'
    
    def test_analyze_company_product_summary_format(self, client):
        """Test analysis with product summary format"""
        payload = {
            'text': '''
                CloudERP is an enterprise resource planning solution that helps
                businesses manage operations, finance, and HR. Key features include
                automated workflows, real-time analytics, and mobile access.
            ''',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['source_type'] == 'product_summary'
    
    def test_analyze_company_whitepaper_format(self, client):
        """Test analysis with whitepaper format"""
        payload = {
            'text': '''
                This whitepaper explores how AI-powered analytics can transform
                business intelligence. Our platform leverages machine learning
                algorithms to provide predictive insights and automated decision-making
                capabilities for enterprise clients.
            ''',
            'source_type': 'whitepaper'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['source_type'] == 'whitepaper'
    
    def test_analyze_company_short_text(self, client):
        """Test analysis with very short text"""
        payload = {
            'text': 'Software company.'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'product_category' in data
    
    def test_analyze_company_long_text(self, client):
        """Test analysis with long text input"""
        long_text = '''
            TechCorp International is a global leader in enterprise software solutions.
            Founded in 2010, the company has grown to serve over 5,000 clients worldwide.
            Our flagship product, CloudERP, revolutionizes how businesses manage their
            operations, finance, and human resources.
        ''' * 50  # Repeat to make it longer
        
        payload = {
            'text': long_text,
            'source_type': 'annual_report'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'product_category' in data
    
    def test_analyze_company_with_special_characters(self, client):
        """Test analysis with special characters"""
        payload = {
            'text': 'TechCorp™ provides AI/ML solutions @ $99.99 per month! Visit www.techcorp.com for more info.'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'product_category' in data
    
    def test_analyze_company_with_unicode(self, client):
        """Test analysis with unicode characters"""
        payload = {
            'text': 'TechCorp提供企业软件解决方案 für Unternehmen en français.'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'product_category' in data
    
    def test_analyze_company_with_numbers(self, client):
        """Test analysis with numerical data"""
        payload = {
            'text': '''
                TechCorp achieved $100M revenue in 2023 with 50% profit margins.
                We serve 1,000+ enterprise clients across 25 countries with 99.99% uptime.
            '''
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'product_category' in data
    
    def test_analyze_company_with_technical_terms(self, client):
        """Test analysis with technical terminology"""
        payload = {
            'text': '''
                Our SaaS platform leverages microservices architecture, Kubernetes
                orchestration, GraphQL APIs, and serverless computing to deliver
                scalable cloud-native solutions with 99.99% uptime SLA.
            '''
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'product_category' in data
    
    def test_analyze_company_with_whitespace(self, client):
        """Test analysis with extra whitespace"""
        payload = {
            'text': '   TechCorp provides software solutions.   \n\n\t  '
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'product_category' in data
    
    # Error Handling Tests
    
    def test_analyze_company_missing_text(self, client):
        """Test analysis with missing text field"""
        payload = {
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'message' in data
    
    def test_analyze_company_empty_text(self, client):
        """Test analysis with empty text"""
        payload = {
            'text': '',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_analyze_company_whitespace_only_text(self, client):
        """Test analysis with whitespace-only text"""
        payload = {
            'text': '   \n\t   ',
            'source_type': 'product_summary'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_analyze_company_invalid_source_type(self, client):
        """Test analysis with invalid source type"""
        payload = {
            'text': 'Some company text',
            'source_type': 'invalid_type'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_analyze_company_no_json(self, client):
        """Test analysis with no JSON body"""
        response = client.post('/api/v1/analyze_company')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_analyze_company_invalid_json(self, client):
        """Test analysis with invalid JSON"""
        response = client.post(
            '/api/v1/analyze_company',
            data='invalid json{',
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_analyze_company_wrong_content_type(self, client):
        """Test analysis with wrong content type"""
        payload = {
            'text': 'Some company text'
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='text/plain'
        )
        
        # Should still work or return appropriate error
        assert response.status_code in [200, 400, 415]
    
    def test_analyze_company_null_text(self, client):
        """Test analysis with null text value"""
        payload = {
            'text': None
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_analyze_company_numeric_text(self, client):
        """Test analysis with numeric text value"""
        payload = {
            'text': 12345
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_analyze_company_array_text(self, client):
        """Test analysis with array text value"""
        payload = {
            'text': ['Some', 'company', 'text']
        }
        
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/api/v1/nonexistent')
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error"""
        response = client.get('/api/v1/analyze_company')
        assert response.status_code == 405
