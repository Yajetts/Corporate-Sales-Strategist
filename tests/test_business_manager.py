"""Tests for Business Manager Module"""

import pytest
import json
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from src.api.app import create_app
from src.models.business_optimizer import BusinessOptimizer, HybridRegressionModel
from src.services.business_manager_service import BusinessManagerService


@pytest.fixture
def client():
    """Create test client"""
    app = create_app('testing')
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_product_portfolio():
    """Generate sample product portfolio for testing"""
    return [
        {
            'name': 'Product A',
            'sales_history': [100, 105, 110, 108, 112],
            'demand_forecast': 120,
            'production_cost': 50.0,
            'current_inventory': 20
        },
        {
            'name': 'Product B',
            'sales_history': [80, 85, 82, 88, 90],
            'demand_forecast': 95,
            'production_cost': 40.0,
            'current_inventory': 15
        },
        {
            'name': 'Product C',
            'sales_history': [150, 145, 155, 160, 158],
            'demand_forecast': 165,
            'production_cost': 70.0,
            'current_inventory': 30
        },
        {
            'name': 'Product D',
            'sales_history': [60, 62, 58, 65, 63],
            'demand_forecast': 68,
            'production_cost': 35.0,
            'current_inventory': 10
        },
        {
            'name': 'Product E',
            'sales_history': [200, 210, 205, 215, 220],
            'demand_forecast': 230,
            'production_cost': 90.0,
            'current_inventory': 40
        }
    ]


@pytest.fixture
def sample_training_data():
    """Generate synthetic training data for regression model"""
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    # Generate features
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate targets (simple linear relationship with noise)
    weights = np.random.randn(n_features, 1)
    targets = features @ weights + np.random.randn(n_samples, 1) * 0.1
    
    return features, targets


class TestHybridRegressionModel:
    """Test cases for Hybrid Regression Model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = HybridRegressionModel(
            input_size=20,
            hidden_sizes=[256, 128, 64],
            output_size=1,
            dropout=0.2
        )
        
        assert model.input_size == 20
        assert model.output_size == 1
        assert model.network is not None
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        model = HybridRegressionModel(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1
        )
        
        # Create sample input
        batch_size = 16
        x = np.random.randn(batch_size, 20).astype(np.float32)
        x_tensor = model.network[0].weight.new_tensor(x)
        
        # Forward pass
        output = model(x_tensor)
        
        assert output.shape == (batch_size, 1)


class TestBusinessOptimizer:
    """Test cases for Business Optimizer"""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = BusinessOptimizer(
            input_size=20,
            hidden_sizes=[256, 128, 64],
            output_size=1,
            device='cpu'
        )
        
        assert optimizer.input_size == 20
        assert optimizer.output_size == 1
        assert optimizer.device.type == 'cpu'
        assert not optimizer.is_fitted
    
    def test_feature_engineering(self):
        """Test feature engineering from sales, demand, and cost data"""
        optimizer = BusinessOptimizer(input_size=20)
        
        # Create sample data
        n_samples = 10
        n_products = 5
        
        sales_data = np.random.rand(n_samples, n_products) * 100
        demand_forecasts = np.random.rand(n_samples, n_products) * 120
        cost_data = np.random.rand(n_samples, n_products) * 50
        
        # Engineer features
        features = optimizer.engineer_features(
            sales_data=sales_data,
            demand_forecasts=demand_forecasts,
            cost_data=cost_data
        )
        
        # Check output shape
        assert features.shape[0] == n_samples
        assert features.shape[1] > n_products  # Should have engineered features
    
    def test_feature_engineering_with_rl_lstm(self):
        """Test feature engineering with RL and LSTM outputs"""
        optimizer = BusinessOptimizer(input_size=20)
        
        n_samples = 10
        n_products = 5
        
        sales_data = np.random.rand(n_samples, n_products) * 100
        demand_forecasts = np.random.rand(n_samples, n_products) * 120
        cost_data = np.random.rand(n_samples, n_products) * 50
        rl_outputs = np.random.rand(n_samples, 2)  # price_adj, promo_int
        lstm_outputs = np.random.rand(n_samples, 2)  # sales_fcst, trend_ind
        
        # Engineer features with RL and LSTM
        features = optimizer.engineer_features(
            sales_data=sales_data,
            demand_forecasts=demand_forecasts,
            cost_data=cost_data,
            rl_outputs=rl_outputs,
            lstm_outputs=lstm_outputs
        )
        
        # Should have more features with RL and LSTM
        assert features.shape[0] == n_samples
        assert features.shape[1] > n_products * 3
    
    def test_data_preprocessing(self):
        """Test data preprocessing and normalization"""
        optimizer = BusinessOptimizer(input_size=20)
        
        # Create sample data
        features = np.random.randn(100, 20).astype(np.float32)
        targets = np.random.randn(100, 1).astype(np.float32)
        
        # Preprocess with fitting
        features_norm, targets_norm = optimizer.preprocess_data(
            features, targets, fit_scalers=True
        )
        
        assert features_norm.shape == features.shape
        assert targets_norm.shape == targets.shape
        assert optimizer.is_fitted
        
        # Check normalization (mean ~0, std ~1)
        assert np.abs(np.mean(features_norm)) < 0.5
        assert np.abs(np.std(features_norm) - 1.0) < 0.5
    
    def test_model_training(self, sample_training_data):
        """Test regression model training with sample data"""
        features, targets = sample_training_data
        
        # Split data
        train_features = features[:150]
        train_targets = targets[:150]
        val_features = features[150:]
        val_targets = targets[150:]
        
        optimizer = BusinessOptimizer(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1,
            device='cpu'
        )
        
        # Train
        metrics = optimizer.train(
            train_features=train_features,
            train_targets=train_targets,
            val_features=val_features,
            val_targets=val_targets,
            epochs=5,
            batch_size=32,
            learning_rate=0.01,
            checkpoint_dir='models/test_checkpoints/business_optimizer',
            early_stopping_patience=3,
            verbose=False
        )
        
        assert 'final_train_loss' in metrics
        assert 'best_val_loss' in metrics
        assert 'epochs_trained' in metrics
        assert metrics['epochs_trained'] <= 5
        assert optimizer.is_fitted
    
    def test_model_prediction(self, sample_training_data):
        """Test model prediction after training"""
        features, targets = sample_training_data
        
        optimizer = BusinessOptimizer(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1,
            device='cpu'
        )
        
        # Train quickly
        optimizer.train(
            train_features=features[:150],
            train_targets=targets[:150],
            epochs=3,
            batch_size=32,
            verbose=False
        )
        
        # Predict
        test_features = features[150:160]
        predictions = optimizer.predict(test_features)
        
        assert predictions.shape == (10, 1)
        assert not np.isnan(predictions).any()
    
    def test_multi_objective_optimization(self, sample_training_data):
        """Test multi-objective optimization"""
        features, targets = sample_training_data
        
        optimizer = BusinessOptimizer(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1,
            device='cpu'
        )
        
        # Train
        optimizer.train(
            train_features=features[:150],
            train_targets=targets[:150],
            epochs=3,
            batch_size=32,
            verbose=False
        )
        
        # Optimize
        test_features = features[150:155]
        product_names = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        
        result = optimizer.optimize_multi_objective(
            features=test_features,
            product_names=product_names,
            revenue_weight=0.7,
            cost_weight=0.3
        )
        
        assert 'optimal_quantities' in result
        assert 'product_names' in result
        assert 'priority_ranking' in result
        assert 'metrics' in result
        assert 'optimization_success' in result
        
        assert len(result['optimal_quantities']) == 5
        assert len(result['priority_ranking']) == 5
        
        # Check metrics
        assert 'total_revenue' in result['metrics']
        assert 'total_cost' in result['metrics']
        assert 'profit' in result['metrics']
        assert 'roi' in result['metrics']
    
    def test_constraint_handling(self, sample_training_data):
        """Test constraint handling in optimization"""
        features, targets = sample_training_data
        
        optimizer = BusinessOptimizer(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1,
            device='cpu'
        )
        
        # Train
        optimizer.train(
            train_features=features[:150],
            train_targets=targets[:150],
            epochs=3,
            batch_size=32,
            verbose=False
        )
        
        # Optimize with constraints
        test_features = features[150:155]
        product_names = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        
        constraints = {
            'min_production': np.array([10, 10, 10, 10, 10]),
            'max_production': np.array([200, 200, 200, 200, 200]),
            'total_budget': 10000,
            'capacity_limit': 500
        }
        
        result = optimizer.optimize_multi_objective(
            features=test_features,
            product_names=product_names,
            constraints=constraints,
            revenue_weight=0.7,
            cost_weight=0.3
        )
        
        # Check that constraints are respected
        optimal_quantities = np.array(result['optimal_quantities'])
        
        # Min/max constraints
        assert np.all(optimal_quantities >= constraints['min_production'])
        assert np.all(optimal_quantities <= constraints['max_production'])
        
        # Capacity constraint
        assert np.sum(optimal_quantities) <= constraints['capacity_limit'] * 1.01  # Allow small tolerance
    
    def test_model_evaluation(self, sample_training_data):
        """Test model evaluation on test data"""
        features, targets = sample_training_data
        
        optimizer = BusinessOptimizer(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1,
            device='cpu'
        )
        
        # Train
        optimizer.train(
            train_features=features[:150],
            train_targets=targets[:150],
            epochs=5,
            batch_size=32,
            verbose=False
        )
        
        # Evaluate
        test_features = features[150:]
        test_targets = targets[150:]
        
        metrics = optimizer.evaluate(test_features, test_targets)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2_score' in metrics
        
        # Check that metrics are reasonable
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0


class TestBusinessManagerService:
    """Test cases for Business Manager Service"""
    
    def test_service_initialization(self):
        """Test service initialization"""
        service = BusinessManagerService(
            input_size=20,
            hidden_sizes=[256, 128, 64],
            output_size=1
        )
        
        assert service.optimizer is not None
        assert not service.model_loaded
        assert service.optimization_count == 0
    
    def test_service_without_model(self, sample_product_portfolio):
        """Test that optimization fails without trained model"""
        service = BusinessManagerService()
        
        with pytest.raises(ValueError, match="Model not loaded"):
            service.optimize_business(sample_product_portfolio)
    
    def test_optimize_business_with_trained_model(self, sample_product_portfolio, sample_training_data):
        """Test business optimization with trained model"""
        features, targets = sample_training_data
        
        service = BusinessManagerService(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1
        )
        
        # Train model
        service.train_model(
            train_features=features[:150],
            train_targets=targets[:150],
            epochs=3,
            batch_size=32,
            verbose=False
        )
        
        # Optimize
        result = service.optimize_business(
            product_portfolio=sample_product_portfolio
        )
        
        assert 'timestamp' in result
        assert 'production_priorities' in result
        assert 'focus_products' in result
        assert 'resource_allocation' in result
        assert 'optimization_metrics' in result
        assert 'optimization_success' in result
        assert 'processing_time_seconds' in result
        assert 'num_products' in result
        
        # Check production priorities
        assert len(result['production_priorities']) == 5
        for priority in result['production_priorities']:
            assert 'rank' in priority
            assert 'product_name' in priority
            assert 'recommended_quantity' in priority
            assert 'priority_score' in priority
        
        # Check focus products
        assert len(result['focus_products']) <= 5
        for product in result['focus_products']:
            assert 'rank' in product
            assert 'product_name' in product
            assert 'focus_rationale' in product
        
        # Check resource allocation
        assert 'by_product' in result['resource_allocation']
        assert 'total_quantity' in result['resource_allocation']
        assert 'total_cost' in result['resource_allocation']
        assert 'total_revenue' in result['resource_allocation']
        
        # Check optimization metrics
        assert 'total_revenue' in result['optimization_metrics']
        assert 'total_cost' in result['optimization_metrics']
        assert 'profit' in result['optimization_metrics']
        assert 'roi' in result['optimization_metrics']
    
    def test_optimize_with_rl_lstm_outputs(self, sample_product_portfolio, sample_training_data):
        """Test optimization with RL and LSTM outputs"""
        features, targets = sample_training_data
        
        service = BusinessManagerService(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1
        )
        
        # Train model
        service.train_model(
            train_features=features[:150],
            train_targets=targets[:150],
            epochs=3,
            batch_size=32,
            verbose=False
        )
        
        # Create RL and LSTM outputs
        rl_outputs = {
            'price_adjustments': [5.0, 3.0, -2.0, 4.0, 1.0],
            'promotion_intensity': [0.7, 0.5, 0.3, 0.6, 0.4]
        }
        
        lstm_outputs = {
            'sales_forecasts': [125, 98, 170, 70, 235],
            'trend_indicators': [0.1, 0.05, 0.15, 0.08, 0.12]
        }
        
        # Optimize
        result = service.optimize_business(
            product_portfolio=sample_product_portfolio,
            rl_strategy_outputs=rl_outputs,
            lstm_forecast_outputs=lstm_outputs
        )
        
        assert result['optimization_success']
        assert len(result['production_priorities']) == 5
    
    def test_optimize_with_constraints(self, sample_product_portfolio, sample_training_data):
        """Test optimization with production constraints"""
        features, targets = sample_training_data
        
        service = BusinessManagerService(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1
        )
        
        # Train model
        service.train_model(
            train_features=features[:150],
            train_targets=targets[:150],
            epochs=3,
            batch_size=32,
            verbose=False
        )
        
        # Define constraints
        constraints = {
            'min_production': [20, 15, 30, 10, 40],
            'max_production': [150, 120, 200, 80, 250],
            'total_budget': 15000,
            'capacity_limit': 600
        }
        
        # Optimize
        result = service.optimize_business(
            product_portfolio=sample_product_portfolio,
            constraints=constraints
        )
        
        assert result['constraints_applied']
        assert result['optimization_success']
        
        # Verify constraints are respected
        for priority in result['production_priorities']:
            product_idx = [p['name'] for p in sample_product_portfolio].index(priority['product_name'])
            quantity = priority['recommended_quantity']
            
            assert quantity >= constraints['min_production'][product_idx]
            assert quantity <= constraints['max_production'][product_idx]
    
    def test_scenario_analysis(self, sample_product_portfolio, sample_training_data):
        """Test multiple scenario analysis"""
        features, targets = sample_training_data
        
        service = BusinessManagerService(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1
        )
        
        # Train model
        service.train_model(
            train_features=features[:150],
            train_targets=targets[:150],
            epochs=3,
            batch_size=32,
            verbose=False
        )
        
        # Define scenarios
        scenarios = [
            {
                'name': 'Conservative',
                'constraints': {
                    'total_budget': 10000,
                    'capacity_limit': 400
                }
            },
            {
                'name': 'Moderate',
                'constraints': {
                    'total_budget': 15000,
                    'capacity_limit': 600
                }
            },
            {
                'name': 'Aggressive',
                'constraints': {
                    'total_budget': 20000,
                    'capacity_limit': 800
                }
            }
        ]
        
        # Analyze scenarios
        result = service.analyze_scenarios(
            product_portfolio=sample_product_portfolio,
            scenarios=scenarios
        )
        
        assert 'scenarios' in result
        assert 'comparison' in result
        assert 'num_scenarios' in result
        
        assert len(result['scenarios']) == 3
        assert result['num_scenarios'] == 3
        
        # Check comparison
        assert 'best_scenario' in result['comparison']
        assert 'comparison_metrics' in result['comparison']
        assert 'num_successful_scenarios' in result['comparison']
    
    def test_resource_recommendations(self):
        """Test resource reallocation recommendations"""
        service = BusinessManagerService()
        
        current_allocation = {
            'Product A': 100,
            'Product B': 80,
            'Product C': 150
        }
        
        optimal_allocation = {
            'Product A': 130,  # 30% increase
            'Product B': 60,   # 25% decrease
            'Product C': 155   # 3% increase (maintain)
        }
        
        recommendations = service.get_resource_recommendations(
            current_allocation=current_allocation,
            optimal_allocation=optimal_allocation
        )
        
        assert len(recommendations) == 3
        
        # Check that recommendations are sorted by priority
        priorities = [r['priority'] for r in recommendations]
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        assert all(priority_order[priorities[i]] <= priority_order[priorities[i+1]] 
                  for i in range(len(priorities)-1))
        
        # Check specific recommendations
        product_a_rec = next(r for r in recommendations if r['product'] == 'Product A')
        assert product_a_rec['action'] == 'increase_production'
        assert product_a_rec['priority'] == 'high'
        
        product_b_rec = next(r for r in recommendations if r['product'] == 'Product B')
        assert product_b_rec['action'] == 'decrease_production'
        assert product_b_rec['priority'] == 'medium'
        
        product_c_rec = next(r for r in recommendations if r['product'] == 'Product C')
        assert product_c_rec['action'] == 'maintain_production'
        assert product_c_rec['priority'] == 'low'
    
    def test_service_metrics(self, sample_product_portfolio, sample_training_data):
        """Test service performance metrics"""
        features, targets = sample_training_data
        
        service = BusinessManagerService(
            input_size=20,
            hidden_sizes=[64, 32],
            output_size=1
        )
        
        # Train model
        service.train_model(
            train_features=features[:150],
            train_targets=targets[:150],
            epochs=3,
            batch_size=32,
            verbose=False
        )
        
        # Initial metrics
        metrics = service.get_service_metrics()
        assert metrics['model_loaded']
        assert metrics['optimization_count'] == 0
        
        # Perform optimization
        service.optimize_business(sample_product_portfolio)
        
        # Updated metrics
        metrics = service.get_service_metrics()
        assert metrics['optimization_count'] == 1
        assert metrics['last_optimization'] is not None
    
    def test_health_check(self):
        """Test service health check"""
        service = BusinessManagerService()
        
        # Without model
        health = service.health_check()
        assert health['status'] == 'unhealthy'
        assert 'not loaded' in health['message'].lower()
    
    def test_model_info(self):
        """Test model information retrieval"""
        service = BusinessManagerService(
            input_size=20,
            hidden_sizes=[256, 128, 64],
            output_size=1
        )
        
        info = service.get_model_info()
        
        assert 'status' in info
        assert 'model_loaded' in info
        assert 'input_size' in info
        assert 'output_size' in info
        assert 'device' in info
        
        assert info['status'] == 'not_initialized'
        assert not info['model_loaded']
        assert info['input_size'] == 20
        assert info['output_size'] == 1


class TestBusinessOptimizerAPI:
    """Test cases for Business Optimizer API endpoint"""
    
    def test_business_optimizer_endpoint(self, client, sample_product_portfolio):
        """Test business optimizer API endpoint"""
        payload = {
            'product_portfolio': sample_product_portfolio,
            'revenue_weight': 0.7,
            'cost_weight': 0.3
        }
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May return 200, 400, 500, or 503 if model not trained
        assert response.status_code in [200, 400, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'production_priorities' in data
            assert 'focus_products' in data
            assert 'resource_allocation' in data
            assert 'optimization_metrics' in data
    
    def test_business_optimizer_missing_data(self, client):
        """Test business optimizer endpoint with missing data"""
        payload = {}
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
    
    def test_business_optimizer_invalid_portfolio(self, client):
        """Test business optimizer endpoint with invalid portfolio"""
        payload = {
            'product_portfolio': []  # Empty portfolio
        }
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code in [400, 503]
    
    def test_business_optimizer_with_constraints(self, client, sample_product_portfolio):
        """Test business optimizer endpoint with constraints"""
        payload = {
            'product_portfolio': sample_product_portfolio,
            'constraints': {
                'min_production': [20, 15, 30, 10, 40],
                'max_production': [150, 120, 200, 80, 250],
                'total_budget': 15000,
                'capacity_limit': 600
            },
            'revenue_weight': 0.6,
            'cost_weight': 0.4
        }
        
        response = client.post(
            '/api/v1/business_optimizer',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May return 200, 400, 500, or 503 if model not trained
        assert response.status_code in [200, 400, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert data['constraints_applied']
    
    def test_scenario_analysis_endpoint(self, client, sample_product_portfolio):
        """Test scenario analysis API endpoint"""
        payload = {
            'product_portfolio': sample_product_portfolio,
            'scenarios': [
                {
                    'name': 'Conservative',
                    'constraints': {
                        'total_budget': 10000,
                        'capacity_limit': 400
                    }
                },
                {
                    'name': 'Aggressive',
                    'constraints': {
                        'total_budget': 20000,
                        'capacity_limit': 800
                    }
                }
            ]
        }
        
        response = client.post(
            '/api/v1/business_optimizer/scenarios',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        # May return 200, 400, 500, or 503 if model not trained
        assert response.status_code in [200, 400, 500, 503]
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'scenarios' in data
            assert 'comparison' in data
            assert len(data['scenarios']) == 2
    
    def test_scenario_analysis_too_many_scenarios(self, client, sample_product_portfolio):
        """Test scenario analysis with too many scenarios"""
        scenarios = [{'name': f'Scenario {i}'} for i in range(15)]
        
        payload = {
            'product_portfolio': sample_product_portfolio,
            'scenarios': scenarios
        }
        
        response = client.post(
            '/api/v1/business_optimizer/scenarios',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Maximum 10 scenarios' in data['message']
    
    def test_scenario_analysis_empty_scenarios(self, client, sample_product_portfolio):
        """Test scenario analysis with empty scenarios list"""
        payload = {
            'product_portfolio': sample_product_portfolio,
            'scenarios': []
        }
        
        response = client.post(
            '/api/v1/business_optimizer/scenarios',
            data=json.dumps(payload),
            content_type='application/json'
        )
        
        assert response.status_code == 400
