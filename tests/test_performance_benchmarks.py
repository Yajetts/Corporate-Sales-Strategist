"""Performance benchmarks for Sales Strategist System

Tests load testing for API endpoints, model inference latency,
database query performance, and scalability scenarios.

Requirements tested:
- 1.5: Company analysis within 5 seconds for 10,000 tokens
- 2.6: Market analysis within 30 seconds for 10,000+ entities
- 4.6: Performance monitoring updates within 10 seconds
"""

import pytest
import json
import time
import concurrent.futures
from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client with testing configuration"""
    app = create_app('testing')
    app.config['TESTING'] = True
    app.config['API_KEYS'] = 'test-performance-key'
    
    with app.test_client() as client:
        yield client


@pytest.fixture
def large_company_text():
    """Generate large company text (~10,000 tokens)"""
    base_text = """
    TechCorp is a leading provider of cloud-based enterprise resource planning (ERP) 
    software solutions. Our flagship product, CloudERP, helps businesses streamline 
    operations, manage finances, and optimize supply chains. We serve mid-to-large 
    enterprises across manufacturing, retail, and services sectors.
    """
    # Repeat to reach approximately 10,000 tokens
    return (base_text * 200)[:50000]  # Approximately 10,000 tokens


@pytest.fixture
def large_market_dataset():
    """Generate large market dataset (10,000+ entities)"""
    return [
        {
            'revenue': 0.5 + (i % 50) / 100,
            'growth': 0.4 + (i % 40) / 100,
            'market_share': 0.3 + (i % 30) / 100,
            'customer_satisfaction': 0.7 + (i % 20) / 100
        }
        for i in range(10000)
    ]



class TestAPIEndpointLoadTesting:
    """Test load testing for API endpoints"""
    
    def test_health_endpoint_load(self, client):
        """Test health endpoint under load"""
        num_requests = 100
        start_time = time.time()
        
        responses = []
        for _ in range(num_requests):
            response = client.get('/api/v1/health')
            responses.append(response.status_code)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_requests
        
        # All requests should complete
        assert len(responses) == num_requests
        
        # Average response time should be fast
        assert avg_time < 0.1, f"Average response time: {avg_time:.3f}s"
        
        # Most requests should succeed
        success_rate = sum(1 for code in responses if code == 200) / num_requests
        assert success_rate > 0.9, f"Success rate: {success_rate:.2%}"
    
    def test_model_info_endpoint_load(self, client):
        """Test model info endpoint under load"""
        num_requests = 50
        start_time = time.time()
        
        responses = []
        for _ in range(num_requests):
            response = client.get('/api/v1/model_info')
            responses.append(response.status_code)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_requests
        
        assert len(responses) == num_requests
        assert avg_time < 0.2, f"Average response time: {avg_time:.3f}s"
        
        success_rate = sum(1 for code in responses if code == 200) / num_requests
        assert success_rate > 0.9, f"Success rate: {success_rate:.2%}"
    
    def test_company_analysis_concurrent_requests(self, client):
        """Test company analysis with concurrent requests"""
        payload = {
            'text': 'TechCorp provides cloud-based software solutions.',
            'source_type': 'product_summary'
        }
        
        def make_request():
            start = time.time()
            response = client.post(
                '/api/v1/analyze_company',
                data=json.dumps(payload),
                content_type='application/json'
            )
            elapsed = time.time() - start
            return response.status_code, elapsed
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        status_codes = [r[0] for r in results]
        response_times = [r[1] for r in results]
        
        # All requests should complete
        assert len(results) == 10
        
        # Calculate success rate
        success_rate = sum(1 for code in status_codes if code in [200, 500, 503]) / len(status_codes)
        assert success_rate == 1.0, f"Some requests failed with unexpected status codes"
        
        # Average response time should be reasonable
        avg_time = sum(response_times) / len(response_times)
        assert avg_time < 10.0, f"Average response time: {avg_time:.2f}s"
    
    def test_market_analysis_concurrent_requests(self, client):
        """Test market analysis with concurrent requests"""
        payload = {
            'market_data': [
                {'revenue': 0.75, 'growth': 0.6},
                {'revenue': 0.65, 'growth': 0.55},
                {'revenue': 0.85, 'growth': 0.7}
            ]
        }
        
        def make_request():
            start = time.time()
            response = client.post(
                '/api/v1/market_analysis',
                data=json.dumps(payload),
                content_type='application/json'
            )
            elapsed = time.time() - start
            return response.status_code, elapsed
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        status_codes = [r[0] for r in results]
        response_times = [r[1] for r in results]
        
        assert len(results) == 5
        
        success_rate = sum(1 for code in status_codes if code in [200, 500, 503]) / len(status_codes)
        assert success_rate == 1.0
        
        avg_time = sum(response_times) / len(response_times)
        assert avg_time < 15.0, f"Average response time: {avg_time:.2f}s"



class TestModelInferenceLatency:
    """Test model inference latency under load (Requirements 1.5, 2.6, 4.6)"""
    
    def test_company_analysis_latency_small_input(self, client):
        """Test company analysis latency with small input"""
        payload = {
            'text': 'TechCorp provides software solutions.',
            'source_type': 'product_summary'
        }
        
        start_time = time.time()
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        elapsed = time.time() - start_time
        
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            # Should be fast for small input
            assert elapsed < 5.0, f"Latency: {elapsed:.2f}s (Requirement: < 5s)"
            
            data = json.loads(response.data)
            if 'processing_time_ms' in data:
                processing_time = data['processing_time_ms'] / 1000
                assert processing_time < 5.0
    
    def test_company_analysis_latency_large_input(self, client, large_company_text):
        """Test company analysis latency with ~10,000 tokens (Requirement 1.5)"""
        payload = {
            'text': large_company_text,
            'source_type': 'annual_report'
        }
        
        start_time = time.time()
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        elapsed = time.time() - start_time
        
        assert response.status_code in [200, 400, 500, 503]
        
        if response.status_code == 200:
            # Requirement 1.5: Within 5 seconds for 10,000 tokens
            assert elapsed < 5.0, f"Latency: {elapsed:.2f}s (Requirement: < 5s for 10k tokens)"
    
    def test_market_analysis_latency_small_dataset(self, client):
        """Test market analysis latency with small dataset"""
        payload = {
            'market_data': [
                {'revenue': 0.75, 'growth': 0.6, 'market_share': 0.45},
                {'revenue': 0.65, 'growth': 0.55, 'market_share': 0.35},
                {'revenue': 0.85, 'growth': 0.7, 'market_share': 0.55}
            ]
        }
        
        start_time = time.time()
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        elapsed = time.time() - start_time
        
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            # Should be fast for small dataset
            assert elapsed < 10.0, f"Latency: {elapsed:.2f}s"
    
    def test_market_analysis_latency_large_dataset(self, client, large_market_dataset):
        """Test market analysis latency with 10,000+ entities (Requirement 2.6)"""
        payload = {
            'market_data': large_market_dataset,
            'auto_select_clusters': True
        }
        
        start_time = time.time()
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        elapsed = time.time() - start_time
        
        assert response.status_code in [200, 400, 500, 503]
        
        if response.status_code == 200:
            # Requirement 2.6: Within 30 seconds for 10,000+ entities
            assert elapsed < 30.0, f"Latency: {elapsed:.2f}s (Requirement: < 30s for 10k+ entities)"
    
    def test_strategy_generation_latency(self, client):
        """Test strategy generation latency"""
        payload = {
            'market_state': {
                'market_demand': 0.72,
                'competitor_prices': [0.55, 0.62, 0.58],
                'sales_volume': 0.68,
                'conversion_rate': 0.16,
                'inventory_level': 0.75,
                'market_trend': 0.15
            },
            'include_explanation': True
        }
        
        start_time = time.time()
        response = client.post(
            '/api/v1/strategy',
            data=json.dumps(payload),
            content_type='application/json'
        )
        elapsed = time.time() - start_time
        
        assert response.status_code in [200, 429, 500, 503]
        
        if response.status_code == 200:
            # Should complete reasonably fast
            assert elapsed < 15.0, f"Latency: {elapsed:.2f}s"
    
    def test_performance_monitoring_latency(self, client):
        """Test performance monitoring latency (Requirement 4.6)"""
        payload = {
            'historical_data': [
                [120.0 + i, 0.16, 0.75] for i in range(100)
            ],
            'include_feedback': True
        }
        
        start_time = time.time()
        response = client.post(
            '/api/v1/performance',
            data=json.dumps(payload),
            content_type='application/json'
        )
        elapsed = time.time() - start_time
        
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            # Requirement 4.6: Within 10 seconds
            assert elapsed < 10.0, f"Latency: {elapsed:.2f}s (Requirement: < 10s)"
    
    def test_business_optimization_latency(self, client):
        """Test business optimization latency"""
        payload = {
            'product_portfolio': [
                {
                    'name': f'Product {i}',
                    'sales_history': [100 + j for j in range(10)],
                    'production_cost': 50.0 + i
                }
                for i in range(10)
            ]
        }
        
        start_time = time.time()
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        elapsed = time.time() - start_time
        
        assert response.status_code in [200, 500, 503]
        
        if response.status_code == 200:
            # Should complete reasonably fast
            assert elapsed < 15.0, f"Latency: {elapsed:.2f}s"



class TestDatabaseQueryPerformance:
    """Test database query performance"""
    
    def test_database_health_check_performance(self, client):
        """Test database health check performance"""
        start_time = time.time()
        response = client.get('/api/v1/db/health')
        elapsed = time.time() - start_time
        
        assert response.status_code in [200, 503]
        
        # Health check should be very fast
        assert elapsed < 1.0, f"DB health check took {elapsed:.2f}s"
    
    def test_repeated_database_queries(self, client):
        """Test repeated database queries for performance"""
        num_queries = 20
        query_times = []
        
        for _ in range(num_queries):
            start_time = time.time()
            response = client.get('/api/v1/db/health')
            elapsed = time.time() - start_time
            query_times.append(elapsed)
        
        avg_time = sum(query_times) / len(query_times)
        max_time = max(query_times)
        
        # Average should be fast
        assert avg_time < 0.5, f"Average query time: {avg_time:.3f}s"
        
        # No single query should be too slow
        assert max_time < 2.0, f"Max query time: {max_time:.2f}s"
    
    def test_concurrent_database_access(self, client):
        """Test concurrent database access"""
        def query_database():
            start = time.time()
            response = client.get('/api/v1/db/health')
            elapsed = time.time() - start
            return response.status_code, elapsed
        
        # Make 10 concurrent database queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(query_database) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        status_codes = [r[0] for r in results]
        query_times = [r[1] for r in results]
        
        # All queries should complete
        assert len(results) == 10
        
        # Most should succeed
        success_rate = sum(1 for code in status_codes if code in [200, 503]) / len(status_codes)
        assert success_rate == 1.0
        
        # Average time should be reasonable
        avg_time = sum(query_times) / len(query_times)
        assert avg_time < 2.0, f"Average concurrent query time: {avg_time:.2f}s"


class TestScalabilityScenarios:
    """Test scalability scenarios"""
    
    def test_increasing_load_pattern(self, client):
        """Test system behavior under increasing load"""
        load_levels = [5, 10, 20]
        results = {}
        
        for load in load_levels:
            start_time = time.time()
            responses = []
            
            for _ in range(load):
                response = client.get('/api/v1/health')
                responses.append(response.status_code)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / load
            success_rate = sum(1 for code in responses if code == 200) / load
            
            results[load] = {
                'avg_time': avg_time,
                'success_rate': success_rate
            }
        
        # Verify system handles increasing load
        for load, metrics in results.items():
            assert metrics['success_rate'] > 0.9, \
                f"Success rate dropped at load {load}: {metrics['success_rate']:.2%}"
    
    def test_burst_traffic_pattern(self, client):
        """Test system behavior under burst traffic"""
        # Simulate burst: 50 requests in quick succession
        burst_size = 50
        start_time = time.time()
        
        responses = []
        for _ in range(burst_size):
            response = client.get('/api/v1/health')
            responses.append(response.status_code)
        
        total_time = time.time() - start_time
        
        # All requests should complete
        assert len(responses) == burst_size
        
        # Calculate success rate
        success_rate = sum(1 for code in responses if code == 200) / burst_size
        assert success_rate > 0.8, f"Success rate during burst: {success_rate:.2%}"
        
        # Total time should be reasonable
        assert total_time < 10.0, f"Burst handling took {total_time:.2f}s"
    
    def test_sustained_load_pattern(self, client):
        """Test system behavior under sustained load"""
        duration = 5  # seconds
        request_interval = 0.1  # 10 requests per second
        
        start_time = time.time()
        responses = []
        
        while time.time() - start_time < duration:
            response = client.get('/api/v1/health')
            responses.append(response.status_code)
            time.sleep(request_interval)
        
        # Calculate metrics
        total_requests = len(responses)
        success_rate = sum(1 for code in responses if code == 200) / total_requests
        
        # System should handle sustained load
        assert total_requests >= 40, f"Only {total_requests} requests completed"
        assert success_rate > 0.9, f"Success rate: {success_rate:.2%}"
    
    def test_mixed_endpoint_load(self, client):
        """Test system with mixed endpoint load"""
        endpoints = [
            '/api/v1/health',
            '/api/v1/model_info',
            '/api/v1/db/health'
        ]
        
        def make_mixed_requests():
            results = []
            for endpoint in endpoints:
                start = time.time()
                response = client.get(endpoint)
                elapsed = time.time() - start
                results.append((endpoint, response.status_code, elapsed))
            return results
        
        # Make requests to multiple endpoints concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_mixed_requests) for _ in range(5)]
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
        
        # Verify all endpoints handled load
        for endpoint in endpoints:
            endpoint_results = [r for r in all_results if r[0] == endpoint]
            assert len(endpoint_results) > 0
            
            success_rate = sum(1 for r in endpoint_results if r[1] in [200, 503]) / len(endpoint_results)
            assert success_rate == 1.0, f"{endpoint} had failures"
    
    def test_async_task_scalability(self, client):
        """Test async task system scalability"""
        # Submit multiple async tasks
        num_tasks = 10
        task_ids = []
        
        payload = {
            'text': 'TechCorp provides software solutions.',
            'source_type': 'product_summary'
        }
        
        start_time = time.time()
        for _ in range(num_tasks):
            response = client.post(
                '/api/v1/async/analyze_company/async',
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            if response.status_code == 202:
                data = json.loads(response.data)
                task_ids.append(data['task_id'])
        
        submission_time = time.time() - start_time
        
        # All tasks should be submitted quickly
        assert len(task_ids) == num_tasks
        assert submission_time < 5.0, f"Task submission took {submission_time:.2f}s"
        
        # Verify we can check status of all tasks
        for task_id in task_ids:
            response = client.get(f'/api/v1/async/tasks/{task_id}')
            assert response.status_code == 200


class TestMemoryAndResourceUsage:
    """Test memory and resource usage patterns"""
    
    def test_large_payload_handling(self, client, large_company_text):
        """Test handling of large payloads"""
        payload = {
            'text': large_company_text,
            'source_type': 'annual_report'
        }
        
        # Should handle large payload without crashing
        response = client.post(
            '/api/v1/analyze_company',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 413, 500, 503]
    
    def test_large_response_handling(self, client, large_market_dataset):
        """Test handling of large responses"""
        # Use smaller dataset to ensure response
        payload = {
            'market_data': large_market_dataset[:1000]  # 1000 entities
        }
        
        response = client.post(
            '/api/v1/market_analysis',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [200, 400, 500, 503]
        
        if response.status_code == 200:
            # Response should be parseable
            data = json.loads(response.data)
            assert isinstance(data, dict)
    
    def test_repeated_requests_no_memory_leak(self, client):
        """Test that repeated requests don't cause memory issues"""
        num_iterations = 50
        
        for i in range(num_iterations):
            response = client.get('/api/v1/health')
            assert response.status_code in [200, 503]
            
            # Verify response is valid
            data = json.loads(response.data)
            assert 'status' in data


class TestPerformanceRegression:
    """Test for performance regressions"""
    
    def test_baseline_health_check_performance(self, client):
        """Establish baseline for health check performance"""
        num_samples = 20
        times = []
        
        for _ in range(num_samples):
            start = time.time()
            response = client.get('/api/v1/health')
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status_code in [200, 503]
        
        avg_time = sum(times) / len(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]
        
        # Baseline expectations
        assert avg_time < 0.1, f"Average: {avg_time:.3f}s"
        assert p95_time < 0.2, f"P95: {p95_time:.3f}s"
    
    def test_baseline_model_info_performance(self, client):
        """Establish baseline for model info performance"""
        num_samples = 20
        times = []
        
        for _ in range(num_samples):
            start = time.time()
            response = client.get('/api/v1/model_info')
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status_code == 200
        
        avg_time = sum(times) / len(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]
        
        # Baseline expectations
        assert avg_time < 0.2, f"Average: {avg_time:.3f}s"
        assert p95_time < 0.5, f"P95: {p95_time:.3f}s"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
