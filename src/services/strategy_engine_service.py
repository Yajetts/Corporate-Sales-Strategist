"""
Strategy Engine Service

This service integrates the RL agent and LLM explainer to provide
comprehensive sales and pricing strategy recommendations.
"""

import os
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from src.models.strategy_agent import StrategyAgent
from src.models.llm_explainer import LLMExplainer
from src.utils.config import Config


class StrategyEngineService:
    """
    Service for generating sales and pricing strategies.
    Combines RL agent predictions with LLM-generated explanations.
    """
    
    def __init__(
        self,
        agent_model_path: Optional[str] = None,
        llm_provider: str = 'openai',
        llm_api_key: Optional[str] = None,
        num_competitors: int = 5,
        enable_caching: bool = True,
        cache_ttl_minutes: int = 60
    ):
        """
        Initialize the Strategy Engine service.
        
        Args:
            agent_model_path: Path to trained RL agent model
            llm_provider: LLM provider ('openai' or 'anthropic')
            llm_api_key: API key for LLM (if None, reads from env)
            num_competitors: Number of competitors in market
            enable_caching: Whether to cache strategy results
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        # Initialize RL agent
        if agent_model_path is None:
            agent_model_path = os.getenv('STRATEGY_AGENT_MODEL_PATH', 
                                         'models/checkpoints/strategy_agent/final_model.zip')
        
        self.agent = StrategyAgent(
            model_path=agent_model_path if os.path.exists(agent_model_path) else None,
            num_competitors=num_competitors
        )
        
        # Initialize LLM explainer
        try:
            self.llm_explainer = LLMExplainer(
                provider=llm_provider,
                api_key=llm_api_key
            )
            self.llm_available = True
        except Exception as e:
            print(f"Warning: LLM explainer initialization failed: {e}")
            print("Strategy explanations will use fallback mode.")
            self.llm_explainer = None
            self.llm_available = False
        
        # Caching
        self.enable_caching = enable_caching
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.cache = {}
    
    def generate_strategy(
        self,
        market_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        include_explanation: bool = True,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive sales and pricing strategy.
        
        Args:
            market_state: Current market conditions
                - market_demand: float (0-1)
                - competitor_prices: list of floats (0-1 normalized)
                - sales_volume: float (0-1)
                - conversion_rate: float (0-1)
                - inventory_level: float (0-1)
                - market_trend: float (-1 to 1)
            context: Additional context (company info, historical data, etc.)
            include_explanation: Whether to generate LLM explanation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Strategy dictionary with recommendations and explanations
        """
        # Check cache
        if self.enable_caching:
            cache_key = self._generate_cache_key(market_state, context)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # Validate market state
        self._validate_market_state(market_state)
        
        # Get strategy from RL agent
        action, strategy_info = self.agent.predict_strategy(
            market_state=market_state,
            deterministic=deterministic
        )
        
        # Calculate confidence score
        confidence = strategy_info['confidence']
        
        # Build strategy response
        strategy = {
            'timestamp': datetime.now().isoformat(),
            'market_state': market_state,
            'recommendations': {
                'price_adjustment_pct': strategy_info['price_adjustment_pct'],
                'sales_approach': strategy_info['sales_approach'],
                'promotion_intensity': strategy_info['promotion_intensity']
            },
            'confidence_score': confidence,
            'confidence_level': self._get_confidence_level(confidence)
        }
        
        # Add explanation if requested
        if include_explanation:
            explanation = self._generate_explanation(
                market_state=market_state,
                strategy_info=strategy_info,
                context=context
            )
            strategy['explanation'] = explanation
        
        # Add actionable insights
        strategy['actionable_insights'] = self._generate_insights(
            market_state=market_state,
            strategy_info=strategy_info
        )
        
        # Cache result
        if self.enable_caching:
            self._add_to_cache(cache_key, strategy)
        
        return strategy
    
    def compare_strategies(
        self,
        market_states: list[Dict[str, Any]],
        scenario_names: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare strategies across multiple market scenarios.
        
        Args:
            market_states: List of market state dictionaries
            scenario_names: Optional names for each scenario
            
        Returns:
            Comparison results with recommendations for each scenario
        """
        if scenario_names is None:
            scenario_names = [f"Scenario {i+1}" for i in range(len(market_states))]
        
        strategies = []
        for market_state in market_states:
            strategy = self.generate_strategy(
                market_state=market_state,
                include_explanation=False
            )
            strategies.append(strategy)
        
        # Analyze differences
        comparison = {
            'scenarios': [],
            'summary': self._summarize_comparison(strategies, scenario_names)
        }
        
        for i, (name, strategy) in enumerate(zip(scenario_names, strategies)):
            comparison['scenarios'].append({
                'name': name,
                'market_state': strategy['market_state'],
                'recommendations': strategy['recommendations'],
                'confidence_score': strategy['confidence_score']
            })
        
        return comparison
    
    def _validate_market_state(self, market_state: Dict[str, Any]):
        """Validate market state input."""
        required_fields = [
            'market_demand', 'competitor_prices', 'sales_volume',
            'conversion_rate', 'inventory_level', 'market_trend'
        ]
        
        for field in required_fields:
            if field not in market_state:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate ranges
        if not 0 <= market_state['market_demand'] <= 1:
            raise ValueError("market_demand must be between 0 and 1")
        
        if not 0 <= market_state['sales_volume'] <= 1:
            raise ValueError("sales_volume must be between 0 and 1")
        
        if not 0 <= market_state['conversion_rate'] <= 1:
            raise ValueError("conversion_rate must be between 0 and 1")
        
        if not 0 <= market_state['inventory_level'] <= 1:
            raise ValueError("inventory_level must be between 0 and 1")
        
        if not -1 <= market_state['market_trend'] <= 1:
            raise ValueError("market_trend must be between -1 and 1")
    
    def _generate_explanation(
        self,
        market_state: Dict[str, Any],
        strategy_info: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate natural language explanation for strategy."""
        if self.llm_available and self.llm_explainer:
            try:
                # Prepare market state with additional info
                enhanced_state = market_state.copy()
                enhanced_state['avg_competitor_price'] = np.mean(
                    market_state.get('competitor_prices', [0.7])
                )
                
                explanation = self.llm_explainer.explain_strategy(
                    market_state=enhanced_state,
                    strategy=strategy_info,
                    context=context
                )
                return explanation
            except Exception as e:
                print(f"LLM explanation failed: {e}. Using fallback.")
        
        # Fallback explanation
        return self.llm_explainer._generate_fallback_explanation(strategy_info) if self.llm_explainer else {
            'summary': 'Strategy generated based on market analysis.',
            'rationale': 'Recommendations optimized for current market conditions.',
            'expected_outcomes': 'Expected to improve overall performance metrics.',
            'risks': 'Monitor market changes and adjust as needed.'
        }
    
    def _generate_insights(
        self,
        market_state: Dict[str, Any],
        strategy_info: Dict[str, Any]
    ) -> list[str]:
        """Generate actionable insights based on strategy."""
        insights = []
        
        price_adj = strategy_info['price_adjustment_pct']
        approach = strategy_info['sales_approach']
        promo = strategy_info['promotion_intensity']
        
        # Price insights
        if price_adj > 5:
            insights.append(f"Consider increasing prices by {price_adj:.1f}% to maximize revenue")
        elif price_adj < -5:
            insights.append(f"Consider decreasing prices by {abs(price_adj):.1f}% to boost sales volume")
        else:
            insights.append("Maintain current pricing levels for optimal balance")
        
        # Sales approach insights
        if approach == 'aggressive':
            insights.append("Adopt aggressive sales tactics: increase outreach, offer time-limited deals")
        elif approach == 'conservative':
            insights.append("Use conservative approach: focus on relationship building and value demonstration")
        else:
            insights.append("Maintain balanced sales approach with steady customer engagement")
        
        # Promotion insights
        if promo > 0.7:
            insights.append(f"Launch high-intensity promotional campaign (intensity: {promo:.0%})")
        elif promo > 0.3:
            insights.append(f"Run moderate promotional activities (intensity: {promo:.0%})")
        else:
            insights.append("Minimal promotion needed; focus on organic growth")
        
        # Market condition insights
        if market_state['market_demand'] < 0.4:
            insights.append("Low market demand detected: consider market expansion or product diversification")
        
        if market_state['conversion_rate'] < 0.15:
            insights.append("Low conversion rate: review sales process and value proposition")
        
        if market_state['inventory_level'] < 0.3:
            insights.append("Low inventory alert: prioritize restocking to avoid stockouts")
        
        if market_state['market_trend'] < -0.2:
            insights.append("Negative market trend: prepare defensive strategies and cost optimization")
        elif market_state['market_trend'] > 0.2:
            insights.append("Positive market trend: capitalize on growth opportunities")
        
        return insights
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to level."""
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _summarize_comparison(
        self,
        strategies: list[Dict[str, Any]],
        scenario_names: list[str]
    ) -> Dict[str, Any]:
        """Summarize strategy comparison."""
        price_adjustments = [s['recommendations']['price_adjustment_pct'] for s in strategies]
        approaches = [s['recommendations']['sales_approach'] for s in strategies]
        confidences = [s['confidence_score'] for s in strategies]
        
        return {
            'price_range': {
                'min': min(price_adjustments),
                'max': max(price_adjustments),
                'avg': np.mean(price_adjustments)
            },
            'most_common_approach': max(set(approaches), key=approaches.count),
            'avg_confidence': np.mean(confidences),
            'highest_confidence_scenario': scenario_names[np.argmax(confidences)]
        }
    
    def _generate_cache_key(
        self,
        market_state: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key from market state and context."""
        # Create deterministic string representation
        state_str = json.dumps(market_state, sort_keys=True)
        context_str = json.dumps(context, sort_keys=True) if context else ""
        combined = state_str + context_str
        
        # Hash to create key
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache if valid."""
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data
            else:
                # Expired, remove from cache
                del self.cache[key]
        return None
    
    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        """Add result to cache."""
        self.cache[key] = (data, datetime.now())
    
    def clear_cache(self):
        """Clear the strategy cache."""
        self.cache.clear()
