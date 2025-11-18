"""
Business Manager Service

This service integrates the hybrid regression model with RL and LSTM outputs
to provide comprehensive manufacturing and resource optimization recommendations.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from src.models.business_optimizer import BusinessOptimizer


class BusinessManagerService:
    """
    Service for manufacturing and resource optimization.
    Combines regression model with RL and LSTM module outputs.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: int = 20,
        hidden_sizes: List[int] = [256, 128, 64],
        output_size: int = 1
    ):
        """
        Initialize Business Manager service.
        
        Args:
            model_path: Path to trained regression model
            input_size: Number of input features
            hidden_sizes: Hidden layer sizes
            output_size: Number of output targets
        """
        # Initialize optimizer
        self.optimizer = BusinessOptimizer(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size
        )
        
        # Load trained model if provided
        if model_path and os.path.exists(model_path):
            self.optimizer.load_checkpoint(model_path)
            self.model_loaded = True
        else:
            self.model_loaded = False
            print("Warning: No trained model loaded. Train model before making predictions.")
        
        # Performance tracking
        self.last_optimization_time = None
        self.optimization_count = 0
    
    def optimize_business(
        self,
        product_portfolio: List[Dict[str, Any]],
        rl_strategy_outputs: Optional[Dict[str, Any]] = None,
        lstm_forecast_outputs: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        revenue_weight: float = 0.7,
        cost_weight: float = 0.3
    ) -> Dict[str, Any]:
        """
        Optimize manufacturing and resource allocation.
        
        Args:
            product_portfolio: List of products with data
                Each product dict should contain:
                - name: Product name
                - sales_history: Historical sales data
                - demand_forecast: Forecasted demand
                - production_cost: Unit production cost
                - current_inventory: Current inventory level
            rl_strategy_outputs: RL agent strategy recommendations (optional)
                - price_adjustments: Price adjustment recommendations per product
                - promotion_intensity: Promotion intensity per product
            lstm_forecast_outputs: LSTM forecast outputs (optional)
                - sales_forecasts: Sales forecasts per product
                - trend_indicators: Trend indicators per product
            constraints: Production constraints (optional)
                - min_production: Minimum production per product
                - max_production: Maximum production per product
                - total_budget: Total budget limit
                - capacity_limit: Total capacity limit
            revenue_weight: Weight for revenue maximization (0-1)
            cost_weight: Weight for cost minimization (0-1)
            
        Returns:
            Optimization results with production priorities and resource allocation
        """
        start_time = time.time()
        
        if not self.model_loaded:
            # Use rule-based fallback for demo purposes
            return self._generate_fallback_optimization(
                product_portfolio=product_portfolio,
                constraints=constraints,
                revenue_weight=revenue_weight,
                cost_weight=cost_weight,
                start_time=start_time
            )
        
        # Validate input
        if not product_portfolio or len(product_portfolio) == 0:
            raise ValueError("product_portfolio cannot be empty")
        
        # Extract product data
        n_products = len(product_portfolio)
        product_names = [p['name'] for p in product_portfolio]
        
        # Build feature matrix
        sales_data = np.array([p.get('sales_history', [0]) for p in product_portfolio])
        if sales_data.ndim == 1:
            sales_data = sales_data.reshape(-1, 1)
        
        demand_forecasts = np.array([p.get('demand_forecast', 0) for p in product_portfolio]).reshape(-1, 1)
        cost_data = np.array([p.get('production_cost', 1.0) for p in product_portfolio]).reshape(-1, 1)
        
        # Integrate RL outputs
        rl_features = None
        if rl_strategy_outputs:
            price_adj = np.array(rl_strategy_outputs.get('price_adjustments', [0] * n_products)).reshape(-1, 1)
            promo_int = np.array(rl_strategy_outputs.get('promotion_intensity', [0.5] * n_products)).reshape(-1, 1)
            rl_features = np.concatenate([price_adj, promo_int], axis=1)
        
        # Integrate LSTM outputs
        lstm_features = None
        if lstm_forecast_outputs:
            sales_fcst = np.array(lstm_forecast_outputs.get('sales_forecasts', [0] * n_products)).reshape(-1, 1)
            trend_ind = np.array(lstm_forecast_outputs.get('trend_indicators', [0] * n_products)).reshape(-1, 1)
            lstm_features = np.concatenate([sales_fcst, trend_ind], axis=1)
        
        # Engineer features
        features = self.optimizer.engineer_features(
            sales_data=sales_data,
            demand_forecasts=demand_forecasts,
            cost_data=cost_data,
            rl_outputs=rl_features,
            lstm_outputs=lstm_features
        )
        
        # Perform multi-objective optimization
        optimization_result = self.optimizer.optimize_multi_objective(
            features=features,
            product_names=product_names,
            constraints=constraints,
            revenue_weight=revenue_weight,
            cost_weight=cost_weight
        )
        
        # Build resource allocation matrix
        resource_matrix = self._build_resource_matrix(
            product_portfolio=product_portfolio,
            optimal_quantities=optimization_result['optimal_quantities']
        )
        
        # Generate production priorities
        production_priorities = self._generate_production_priorities(
            product_names=product_names,
            optimal_quantities=optimization_result['optimal_quantities'],
            priority_ranking=optimization_result['priority_ranking'],
            product_portfolio=product_portfolio
        )
        
        # Calculate focus products (top performers)
        focus_products = self._identify_focus_products(
            production_priorities=production_priorities,
            top_n=min(5, n_products)
        )
        
        # Build response
        processing_time = time.time() - start_time
        self.optimization_count += 1
        self.last_optimization_time = datetime.now()
        
        response = {
            'timestamp': datetime.now().isoformat(),
            'production_priorities': production_priorities,
            'focus_products': focus_products,
            'resource_allocation': resource_matrix,
            'optimization_metrics': optimization_result['metrics'],
            'constraints_applied': constraints is not None,
            'optimization_success': optimization_result['optimization_success'],
            'processing_time_seconds': processing_time,
            'num_products': n_products
        }
        
        return response
    
    def analyze_scenarios(
        self,
        product_portfolio: List[Dict[str, Any]],
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze multiple what-if scenarios.
        
        Args:
            product_portfolio: Base product portfolio
            scenarios: List of scenario configurations
                Each scenario should contain:
                - name: Scenario name
                - constraints: Constraint modifications
                - rl_outputs: RL output modifications (optional)
                - lstm_outputs: LSTM output modifications (optional)
                
        Returns:
            Scenario comparison results
        """
        if not scenarios or len(scenarios) == 0:
            raise ValueError("scenarios cannot be empty")
        
        if len(scenarios) > 10:
            raise ValueError("Maximum 10 scenarios allowed")
        
        scenario_results = []
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'Unnamed Scenario')
            constraints = scenario.get('constraints')
            rl_outputs = scenario.get('rl_outputs')
            lstm_outputs = scenario.get('lstm_outputs')
            
            try:
                result = self.optimize_business(
                    product_portfolio=product_portfolio,
                    rl_strategy_outputs=rl_outputs,
                    lstm_forecast_outputs=lstm_outputs,
                    constraints=constraints
                )
                
                scenario_results.append({
                    'name': scenario_name,
                    'production_priorities': result['production_priorities'],
                    'focus_products': result['focus_products'],
                    'metrics': result['optimization_metrics'],
                    'success': result['optimization_success']
                })
            except Exception as e:
                scenario_results.append({
                    'name': scenario_name,
                    'error': str(e),
                    'success': False
                })
        
        # Compare scenarios
        comparison = self._compare_scenarios(scenario_results)
        
        return {
            'scenarios': scenario_results,
            'comparison': comparison,
            'num_scenarios': len(scenarios)
        }
    
    def get_resource_recommendations(
        self,
        current_allocation: Dict[str, float],
        optimal_allocation: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for resource reallocation.
        
        Args:
            current_allocation: Current resource allocation per product
            optimal_allocation: Optimal resource allocation per product
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        for product_name in optimal_allocation:
            current = current_allocation.get(product_name, 0)
            optimal = optimal_allocation[product_name]
            
            if optimal > current * 1.2:  # 20% increase threshold
                recommendations.append({
                    'product': product_name,
                    'action': 'increase_production',
                    'current_quantity': current,
                    'recommended_quantity': optimal,
                    'change_pct': ((optimal - current) / current * 100) if current > 0 else 100,
                    'priority': 'high',
                    'rationale': f'Demand forecast indicates {optimal:.0f} units needed vs current {current:.0f}'
                })
            elif optimal < current * 0.8:  # 20% decrease threshold
                recommendations.append({
                    'product': product_name,
                    'action': 'decrease_production',
                    'current_quantity': current,
                    'recommended_quantity': optimal,
                    'change_pct': ((optimal - current) / current * 100) if current > 0 else -100,
                    'priority': 'medium',
                    'rationale': f'Reduce production to {optimal:.0f} units from current {current:.0f} to optimize costs'
                })
            else:
                recommendations.append({
                    'product': product_name,
                    'action': 'maintain_production',
                    'current_quantity': current,
                    'recommended_quantity': optimal,
                    'change_pct': ((optimal - current) / current * 100) if current > 0 else 0,
                    'priority': 'low',
                    'rationale': 'Current production levels are near optimal'
                })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def _build_resource_matrix(
        self,
        product_portfolio: List[Dict[str, Any]],
        optimal_quantities: List[float]
    ) -> Dict[str, Any]:
        """Build resource allocation matrix."""
        total_quantity = sum(optimal_quantities)
        
        allocation = {}
        for i, product in enumerate(product_portfolio):
            product_name = product['name']
            quantity = optimal_quantities[i]
            
            allocation[product_name] = {
                'quantity': float(quantity),
                'percentage': float(quantity / total_quantity * 100) if total_quantity > 0 else 0,
                'estimated_cost': float(quantity * product.get('production_cost', 1.0)),
                'estimated_revenue': float(quantity * product.get('demand_forecast', 0))
            }
        
        return {
            'by_product': allocation,
            'total_quantity': float(total_quantity),
            'total_cost': sum(a['estimated_cost'] for a in allocation.values()),
            'total_revenue': sum(a['estimated_revenue'] for a in allocation.values())
        }
    
    def _generate_production_priorities(
        self,
        product_names: List[str],
        optimal_quantities: List[float],
        priority_ranking: List[str],
        product_portfolio: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate production priority list."""
        priorities = []
        
        for rank, product_name in enumerate(priority_ranking):
            idx = product_names.index(product_name)
            product_data = product_portfolio[idx]
            
            priorities.append({
                'rank': rank + 1,
                'product_name': product_name,
                'recommended_quantity': float(optimal_quantities[idx]),
                'demand_forecast': product_data.get('demand_forecast', 0),
                'production_cost': product_data.get('production_cost', 1.0),
                'priority_score': float(optimal_quantities[idx] * product_data.get('demand_forecast', 0))
            })
        
        return priorities
    
    def _identify_focus_products(
        self,
        production_priorities: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify top focus products."""
        # Get top N products by priority
        focus = production_priorities[:top_n]
        
        # Add focus rationale
        for product in focus:
            if product['rank'] == 1:
                product['focus_rationale'] = 'Highest priority product with maximum revenue potential'
            elif product['rank'] <= 3:
                product['focus_rationale'] = 'High-priority product with strong demand forecast'
            else:
                product['focus_rationale'] = 'Important product for balanced portfolio'
        
        return focus
    
    def _compare_scenarios(
        self,
        scenario_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple scenarios."""
        successful_scenarios = [s for s in scenario_results if s.get('success', False)]
        
        if not successful_scenarios:
            return {
                'best_scenario': None,
                'comparison_metrics': {},
                'message': 'No successful scenarios to compare'
            }
        
        # Find best scenario by profit
        best_scenario = max(
            successful_scenarios,
            key=lambda s: s['metrics'].get('profit', 0)
        )
        
        # Calculate comparison metrics
        profits = [s['metrics'].get('profit', 0) for s in successful_scenarios]
        revenues = [s['metrics'].get('total_revenue', 0) for s in successful_scenarios]
        rois = [s['metrics'].get('roi', 0) for s in successful_scenarios]
        
        return {
            'best_scenario': best_scenario['name'],
            'best_scenario_profit': best_scenario['metrics'].get('profit', 0),
            'comparison_metrics': {
                'profit_range': {
                    'min': min(profits),
                    'max': max(profits),
                    'avg': np.mean(profits)
                },
                'revenue_range': {
                    'min': min(revenues),
                    'max': max(revenues),
                    'avg': np.mean(revenues)
                },
                'roi_range': {
                    'min': min(rois),
                    'max': max(rois),
                    'avg': np.mean(rois)
                }
            },
            'num_successful_scenarios': len(successful_scenarios)
        }
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        return {
            'model_loaded': self.model_loaded,
            'optimization_count': self.optimization_count,
            'last_optimization': self.last_optimization_time.isoformat() if self.last_optimization_time else None
        }
    
    def health_check(self) -> Dict[str, str]:
        """Health check for the service."""
        if self.model_loaded:
            return {
                'status': 'healthy',
                'message': 'Business Manager service is operational'
            }
        else:
            return {
                'status': 'unhealthy',
                'message': 'Model not loaded'
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'status': 'ready' if self.model_loaded else 'not_initialized',
            'model_loaded': self.model_loaded,
            'input_size': self.optimizer.input_size,
            'output_size': self.optimizer.output_size,
            'device': str(self.optimizer.device)
        }
    
    def train_model(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        **train_kwargs
    ) -> Dict[str, Any]:
        """
        Train the regression model.
        
        Args:
            train_features: Training features
            train_targets: Training targets
            val_features: Validation features
            val_targets: Validation targets
            **train_kwargs: Additional training arguments
            
        Returns:
            Training metrics
        """
        metrics = self.optimizer.train(
            train_features=train_features,
            train_targets=train_targets,
            val_features=val_features,
            val_targets=val_targets,
            **train_kwargs
        )
        
        self.model_loaded = True
        
        return metrics
    
    def evaluate_model(
        self,
        test_features: np.ndarray,
        test_targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_features: Test features
            test_targets: Test targets
            
        Returns:
            Evaluation metrics
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Train or load model first.")
        
        return self.optimizer.evaluate(test_features, test_targets)

    
    def _generate_fallback_optimization(
        self,
        product_portfolio: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]],
        revenue_weight: float,
        cost_weight: float,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Generate rule-based optimization when no trained model is available.
        This provides a reasonable fallback for demo purposes.
        
        Args:
            product_portfolio: List of products with data
            constraints: Production constraints
            revenue_weight: Weight for revenue maximization
            cost_weight: Weight for cost minimization
            start_time: Start time for performance tracking
            
        Returns:
            Optimization results with synthetic recommendations
        """
        # Extract product data
        n_products = len(product_portfolio)
        
        # Calculate profit margin for each product
        recommendations = []
        for product in product_portfolio:
            name = product.get('name', 'Unknown')
            demand = product.get('demand', 1000)
            cost = product.get('cost', 50)
            price = product.get('price', 80)
            min_production = product.get('min_production', 100)
            
            # Calculate profit margin
            profit_margin = (price - cost) / price if price > 0 else 0
            
            # Calculate priority score based on profit margin and demand
            # Higher profit margin and higher demand = higher priority
            normalized_demand = demand / 1000  # Normalize to 0-1 range
            priority_score = (profit_margin * revenue_weight + 
                            (1 - cost / price) * cost_weight + 
                            min(normalized_demand, 1.0) * 0.3)
            priority_score = min(max(priority_score, 0), 1)  # Clip to 0-1
            
            # Calculate optimal quantity based on demand and constraints
            optimal_quantity = demand
            
            # Apply constraints if provided
            if constraints:
                budget_limit = constraints.get('budget_limit', float('inf'))
                capacity_limit = constraints.get('capacity_limit', float('inf'))
                
                # Adjust quantity based on budget
                max_by_budget = budget_limit / cost if cost > 0 else float('inf')
                optimal_quantity = min(optimal_quantity, max_by_budget)
                
                # Adjust quantity based on capacity
                optimal_quantity = min(optimal_quantity, capacity_limit / n_products)
            
            # Ensure minimum production
            optimal_quantity = max(optimal_quantity, min_production)
            
            recommendations.append({
                'name': name,
                'quantity': int(optimal_quantity),
                'demand': demand,
                'cost': cost,
                'price': price,
                'priority_score': priority_score,
                'profit_margin': profit_margin,
                'total_cost': optimal_quantity * cost,
                'total_revenue': optimal_quantity * price,
                'total_profit': optimal_quantity * (price - cost)
            })
        
        # Sort by priority score
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Calculate overall metrics
        total_revenue = sum(r['total_revenue'] for r in recommendations)
        total_cost = sum(r['total_cost'] for r in recommendations)
        total_profit = total_revenue - total_cost
        roi = (total_profit / total_cost) if total_cost > 0 else 0
        
        # Identify focus products (top 5)
        focus_products = [
            {
                'name': r['name'],
                'priority_score': r['priority_score'],
                'recommended_quantity': r['quantity'],
                'expected_revenue': r['total_revenue'],
                'profit_margin': r['profit_margin']
            }
            for r in recommendations[:min(5, n_products)]
        ]
        
        # Build resource allocation matrix
        resource_allocation = {
            'total_quantity': sum(r['quantity'] for r in recommendations),
            'total_cost': total_cost,
            'total_revenue': total_revenue,
            'by_product': [
                {
                    'name': r['name'],
                    'quantity': r['quantity'],
                    'cost': r['total_cost'],
                    'revenue': r['total_revenue']
                }
                for r in recommendations
            ]
        }
        
        # Track processing time
        processing_time = time.time() - start_time
        self.optimization_count += 1
        self.last_optimization_time = datetime.now()
        
        # Build response
        response = {
            'timestamp': datetime.now().isoformat(),
            'production_priorities': recommendations,
            'recommendations': recommendations,  # Also include for backward compatibility
            'focus_products': focus_products,
            'resource_allocation': resource_allocation,
            'optimization_metrics': {
                'total_revenue': total_revenue,
                'total_cost': total_cost,
                'profit': total_profit,
                'roi': roi
            },
            'constraints_applied': constraints is not None,
            'optimization_success': True,
            'processing_time_seconds': processing_time,
            'num_products': n_products
        }
        
        return response
