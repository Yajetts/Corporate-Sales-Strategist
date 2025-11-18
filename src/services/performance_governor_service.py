"""
Performance Governor Service

This service integrates LSTM forecasting, anomaly detection, and feedback loop
to provide comprehensive performance monitoring and strategy optimization.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import time

from src.models.sales_forecaster import SalesForecaster
from src.models.anomaly_detector import AnomalyDetector, SeverityLevel
from src.models.feedback_loop import FeedbackLearningLoop


class PerformanceGovernorService:
    """
    Service for real-time performance monitoring and feedback.
    Combines LSTM forecasting, anomaly detection, and feedback learning.
    """
    
    def __init__(
        self,
        forecaster_model_path: Optional[str] = None,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 7,
        sequence_length: int = 30,
        anomaly_thresholds: Optional[Dict[str, float]] = None,
        feedback_learning_rate: float = 0.1,
        enable_feedback_loop: bool = True
    ):
        """
        Initialize Performance Governor service.
        
        Args:
            forecaster_model_path: Path to trained LSTM model
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            forecast_horizon: Days to forecast
            sequence_length: Input sequence length
            anomaly_thresholds: Custom thresholds for anomaly detection
            feedback_learning_rate: Learning rate for feedback loop
            enable_feedback_loop: Whether to enable feedback learning
        """
        # Initialize LSTM forecaster
        self.forecaster = SalesForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            sequence_length=sequence_length
        )
        
        # Load trained model if provided
        if forecaster_model_path and os.path.exists(forecaster_model_path):
            self.forecaster.load_checkpoint(forecaster_model_path)
            self.model_loaded = True
        else:
            self.model_loaded = False
            print("Warning: No trained model loaded. Train model before making predictions.")
        
        # Initialize anomaly detector
        if anomaly_thresholds is None:
            anomaly_thresholds = {
                'threshold_low': 0.15,
                'threshold_medium': 0.25,
                'threshold_high': 0.40,
                'threshold_critical': 0.60
            }
        
        self.anomaly_detector = AnomalyDetector(**anomaly_thresholds)
        
        # Initialize feedback loop
        self.enable_feedback_loop = enable_feedback_loop
        if enable_feedback_loop:
            self.feedback_loop = FeedbackLearningLoop(
                learning_rate=feedback_learning_rate
            )
        else:
            self.feedback_loop = None
        
        # Performance tracking
        self.last_update_time = None
        self.processing_times = []
    
    def monitor_performance(
        self,
        historical_data: np.ndarray,
        current_data: Optional[np.ndarray] = None,
        strategy_context: Optional[Dict[str, Any]] = None,
        include_feedback: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive performance monitoring with forecasting and anomaly detection.
        
        Args:
            historical_data: Historical time-series data (n_timesteps, n_features)
            current_data: Current/recent data for comparison (optional)
            strategy_context: Context about applied strategies
            include_feedback: Whether to include feedback analysis
            
        Returns:
            Complete performance report with forecasts, alerts, and feedback
        """
        start_time = time.time()
        
        if not self.model_loaded:
            # Use rule-based fallback for demo purposes
            return self._generate_fallback_forecast(
                historical_data=historical_data,
                current_data=current_data,
                strategy_context=strategy_context,
                include_feedback=include_feedback,
                start_time=start_time
            )
        
        # Validate input
        if len(historical_data) < self.forecaster.sequence_length:
            raise ValueError(f"Historical data must have at least {self.forecaster.sequence_length} timesteps")
        
        # Generate forecast
        forecast_result = self.forecaster.predict(
            data=historical_data,
            return_confidence=True
        )
        
        forecast_values = np.array(forecast_result['forecast'])
        confidence_interval = forecast_result.get('confidence_interval', {})
        
        # Detect anomalies if current data provided
        alerts = []
        if current_data is not None and len(current_data) > 0:
            # Compare recent predictions with actual values
            recent_sequence = historical_data[-self.forecaster.sequence_length-len(current_data):-len(current_data)]
            
            if len(recent_sequence) >= self.forecaster.sequence_length:
                recent_forecast = self.forecaster.predict(recent_sequence, return_confidence=True)
                recent_predicted = np.array(recent_forecast['forecast'])[:len(current_data)]
                recent_actual = current_data[:, 0]  # First feature (sales)
                
                # Generate timestamps
                timestamps = [datetime.now() - timedelta(days=len(current_data)-i) for i in range(len(current_data))]
                
                # Detect anomalies
                alerts = self.anomaly_detector.detect_anomalies(
                    actual_values=recent_actual,
                    predicted_values=recent_predicted,
                    confidence_intervals={
                        'lower': np.array(recent_forecast['confidence_interval']['lower'])[:len(current_data)],
                        'upper': np.array(recent_forecast['confidence_interval']['upper'])[:len(current_data)]
                    } if 'confidence_interval' in recent_forecast else None,
                    metric_name='sales',
                    timestamps=timestamps
                )
                
                # Add feedback signals if enabled
                if self.enable_feedback_loop and include_feedback:
                    for i in range(len(recent_actual)):
                        self.feedback_loop.add_feedback_signal(
                            predicted_value=float(recent_predicted[i]),
                            actual_value=float(recent_actual[i]),
                            metric_name='sales',
                            strategy_context=strategy_context,
                            timestamp=timestamps[i]
                        )
        
        # Build response
        response = {
            'timestamp': datetime.now().isoformat(),
            'forecast': {
                'values': forecast_values.tolist(),
                'horizon_days': self.forecaster.forecast_horizon,
                'confidence_interval': {
                    'lower': confidence_interval.get('lower', []),
                    'upper': confidence_interval.get('upper', []),
                    'confidence_level': confidence_interval.get('confidence_level', 0.90)
                } if confidence_interval else None
            },
            'alerts': {
                'total': len(alerts),
                'by_severity': self._count_alerts_by_severity(alerts),
                'details': [alert.to_dict() for alert in alerts]
            },
            'alert_summary': self.anomaly_detector.get_alert_summary(time_window_hours=24)
        }
        
        # Add feedback summary if enabled
        if self.enable_feedback_loop and include_feedback:
            response['feedback'] = {
                'summary': self.feedback_loop.get_feedback_summary(time_window_hours=24),
                'strategy_weights': self.feedback_loop.get_strategy_weights(),
                'rl_weight_adjustments': self.feedback_loop.get_weight_adjustments_for_rl()
            }
        
        # Add trend analysis
        response['trend_analysis'] = self._analyze_trends(
            historical_data=historical_data,
            forecast_values=forecast_values
        )
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        response['processing_time_seconds'] = processing_time
        
        self.last_update_time = datetime.now()
        
        return response
    
    def get_performance_trends(
        self,
        historical_data: np.ndarray,
        window_size: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            historical_data: Historical time-series data
            window_size: Window size for trend calculation
            
        Returns:
            Trend analysis results
        """
        if len(historical_data) < window_size:
            window_size = len(historical_data)
        
        recent_data = historical_data[-window_size:, 0]  # First feature (sales)
        
        # Calculate trend
        x = np.arange(len(recent_data))
        trend_coef = np.polyfit(x, recent_data, 1)[0]
        
        # Calculate volatility
        volatility = np.std(recent_data)
        
        # Calculate moving average
        ma_short = np.mean(recent_data[-7:]) if len(recent_data) >= 7 else np.mean(recent_data)
        ma_long = np.mean(recent_data[-30:]) if len(recent_data) >= 30 else np.mean(recent_data)
        
        # Determine trend direction
        if trend_coef > 0.5:
            trend_direction = 'increasing'
        elif trend_coef < -0.5:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        return {
            'trend_direction': trend_direction,
            'trend_coefficient': float(trend_coef),
            'volatility': float(volatility),
            'moving_average_7d': float(ma_short),
            'moving_average_30d': float(ma_long),
            'current_value': float(recent_data[-1]),
            'period_change_pct': float((recent_data[-1] - recent_data[0]) / recent_data[0] * 100)
        }
    
    def get_critical_alerts(
        self,
        severity_threshold: SeverityLevel = SeverityLevel.HIGH
    ) -> List[Dict[str, Any]]:
        """
        Get critical alerts that require immediate attention.
        
        Args:
            severity_threshold: Minimum severity level
            
        Returns:
            List of critical alerts
        """
        critical_alerts = []
        
        for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            if severity.value >= severity_threshold.value:
                alerts = self.anomaly_detector.get_alerts_by_severity(severity, limit=10)
                critical_alerts.extend(alerts)
        
        # Sort by timestamp (most recent first)
        critical_alerts.sort(key=lambda a: a['timestamp'], reverse=True)
        
        return critical_alerts
    
    def get_feedback_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations based on feedback learning.
        
        Returns:
            Feedback-based recommendations
        """
        if not self.enable_feedback_loop:
            return {'enabled': False, 'message': 'Feedback loop not enabled'}
        
        weights = self.feedback_loop.get_strategy_weights()
        summary = self.feedback_loop.get_feedback_summary(time_window_hours=168)  # 1 week
        
        recommendations = []
        
        # Analyze weights and provide recommendations
        if weights['price_sensitivity'] < 0.8:
            recommendations.append({
                'component': 'pricing',
                'recommendation': 'Consider more conservative pricing strategies',
                'reason': f"Price sensitivity weight is low ({weights['price_sensitivity']:.2f})",
                'priority': 'high'
            })
        elif weights['price_sensitivity'] > 1.3:
            recommendations.append({
                'component': 'pricing',
                'recommendation': 'Pricing strategies are performing well, consider testing higher prices',
                'reason': f"Price sensitivity weight is high ({weights['price_sensitivity']:.2f})",
                'priority': 'medium'
            })
        
        if weights['promotion_effectiveness'] < 0.8:
            recommendations.append({
                'component': 'promotion',
                'recommendation': 'Review promotional strategies and targeting',
                'reason': f"Promotion effectiveness is low ({weights['promotion_effectiveness']:.2f})",
                'priority': 'high'
            })
        elif weights['promotion_effectiveness'] > 1.3:
            recommendations.append({
                'component': 'promotion',
                'recommendation': 'Promotions are highly effective, consider increasing investment',
                'reason': f"Promotion effectiveness is high ({weights['promotion_effectiveness']:.2f})",
                'priority': 'medium'
            })
        
        if weights['sales_approach_impact'] < 0.8:
            recommendations.append({
                'component': 'sales_approach',
                'recommendation': 'Adjust sales approach or provide additional training',
                'reason': f"Sales approach impact is low ({weights['sales_approach_impact']:.2f})",
                'priority': 'high'
            })
        
        if summary['avg_performance_score'] < 0.6:
            recommendations.append({
                'component': 'overall',
                'recommendation': 'Overall performance below target, consider strategic review',
                'reason': f"Average performance score is {summary['avg_performance_score']:.2f}",
                'priority': 'critical'
            })
        
        return {
            'enabled': True,
            'current_weights': weights,
            'performance_summary': summary,
            'recommendations': recommendations,
            'recent_adjustments': self.feedback_loop.get_adjustment_history(limit=5)
        }
    
    def _analyze_trends(
        self,
        historical_data: np.ndarray,
        forecast_values: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze trends in historical and forecasted data."""
        # Historical trend
        hist_sales = historical_data[-30:, 0] if len(historical_data) >= 30 else historical_data[:, 0]
        hist_trend = np.polyfit(range(len(hist_sales)), hist_sales, 1)[0]
        
        # Forecast trend
        forecast_trend = np.polyfit(range(len(forecast_values)), forecast_values, 1)[0]
        
        # Compare trends
        if hist_trend > 0 and forecast_trend > 0:
            trend_outlook = 'positive'
        elif hist_trend < 0 and forecast_trend < 0:
            trend_outlook = 'negative'
        elif hist_trend < 0 and forecast_trend > 0:
            trend_outlook = 'recovering'
        elif hist_trend > 0 and forecast_trend < 0:
            trend_outlook = 'declining'
        else:
            trend_outlook = 'stable'
        
        return {
            'historical_trend': float(hist_trend),
            'forecast_trend': float(forecast_trend),
            'trend_outlook': trend_outlook,
            'forecast_vs_current_pct': float((forecast_values[0] - hist_sales[-1]) / hist_sales[-1] * 100)
        }
    
    def _count_alerts_by_severity(self, alerts: List) -> Dict[str, int]:
        """Count alerts by severity level."""
        counts = {severity.value: 0 for severity in SeverityLevel}
        
        for alert in alerts:
            counts[alert.severity.value] += 1
        
        return counts
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        if not self.processing_times:
            return {
                'total_predictions': 0,
                'avg_processing_time': 0.0,
                'last_update': None
            }
        
        return {
            'total_predictions': len(self.processing_times),
            'avg_processing_time': float(np.mean(self.processing_times)),
            'min_processing_time': float(np.min(self.processing_times)),
            'max_processing_time': float(np.max(self.processing_times)),
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'model_loaded': self.model_loaded,
            'feedback_enabled': self.enable_feedback_loop
        }
    
    def train_forecaster(
        self,
        train_data: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        **train_kwargs
    ) -> Dict[str, Any]:
        """
        Train the LSTM forecaster.
        
        Args:
            train_data: Training data
            val_data: Validation data
            **train_kwargs: Additional training arguments
            
        Returns:
            Training metrics
        """
        metrics = self.forecaster.train(
            train_data=train_data,
            val_data=val_data,
            **train_kwargs
        )
        
        self.model_loaded = True
        
        return metrics
    
    def evaluate_forecaster(
        self,
        test_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate forecaster on test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Evaluation metrics
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Train or load model first.")
        
        return self.forecaster.evaluate(test_data)
    
    def _generate_fallback_forecast(
        self,
        historical_data: np.ndarray,
        current_data: Optional[np.ndarray],
        strategy_context: Optional[Dict[str, Any]],
        include_feedback: bool,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Generate rule-based forecast when no trained model is available.
        This provides a reasonable fallback for demo purposes.
        
        Args:
            historical_data: Historical time-series data
            current_data: Current/recent data for comparison
            strategy_context: Context about applied strategies
            include_feedback: Whether to include feedback analysis
            start_time: Start time for performance tracking
            
        Returns:
            Forecast results with synthetic predictions
        """
        # Extract sales data (first feature)
        sales_data = historical_data[:, 0]
        
        # Calculate trend using simple linear regression
        x = np.arange(len(sales_data))
        trend_coef = np.polyfit(x, sales_data, 1)[0]
        intercept = np.polyfit(x, sales_data, 1)[1]
        
        # Calculate seasonality (simple moving average deviation)
        window_size = min(7, len(sales_data))
        ma = np.convolve(sales_data, np.ones(window_size)/window_size, mode='valid')
        
        # Generate forecast
        last_value = sales_data[-1]
        forecast_values = []
        
        for i in range(self.forecaster.forecast_horizon):
            # Trend component
            trend_value = trend_coef * (len(sales_data) + i) + intercept
            
            # Add some noise for realism
            noise = np.random.normal(0, np.std(sales_data) * 0.1)
            
            # Combine components
            forecast_value = trend_value + noise
            
            # Ensure non-negative
            forecast_value = max(0, forecast_value)
            
            forecast_values.append(forecast_value)
        
        forecast_values = np.array(forecast_values)
        
        # Generate confidence intervals (±20% of forecast)
        confidence_margin = forecast_values * 0.2
        lower_bound = forecast_values - confidence_margin
        upper_bound = forecast_values + confidence_margin
        
        # Detect anomalies if current data provided
        alerts = []
        if current_data is not None and len(current_data) > 0:
            # Simple anomaly detection: values outside 2 std devs
            mean_sales = np.mean(sales_data)
            std_sales = np.std(sales_data)
            
            for i, value in enumerate(current_data[:, 0]):
                deviation = abs(value - mean_sales) / std_sales
                
                if deviation > 2.5:
                    severity = SeverityLevel.CRITICAL
                elif deviation > 2.0:
                    severity = SeverityLevel.HIGH
                elif deviation > 1.5:
                    severity = SeverityLevel.MEDIUM
                else:
                    continue
                
                timestamp = datetime.now() - timedelta(days=len(current_data)-i)
                
                alert = type('Alert', (), {
                    'alert_id': f'fallback_alert_{i}',
                    'timestamp': timestamp,
                    'anomaly_type': 'statistical_deviation',
                    'severity': severity,
                    'metric_name': 'sales',
                    'actual_value': float(value),
                    'expected_value': float(mean_sales),
                    'deviation_pct': float(deviation * 100),
                    'description': f'Sales value deviates {deviation:.1f} standard deviations from mean',
                    'recommended_actions': [
                        'Review recent market changes',
                        'Analyze competitor activity',
                        'Check data quality'
                    ],
                    'to_dict': lambda self: {
                        'alert_id': self.alert_id,
                        'timestamp': self.timestamp.isoformat(),
                        'anomaly_type': self.anomaly_type,
                        'severity': self.severity.value,
                        'metric_name': self.metric_name,
                        'actual_value': self.actual_value,
                        'expected_value': self.expected_value,
                        'deviation_pct': self.deviation_pct,
                        'description': self.description,
                        'recommended_actions': self.recommended_actions
                    }
                })()
                
                alerts.append(alert)
        
        # Build response
        response = {
            'timestamp': datetime.now().isoformat(),
            'forecast': {
                'values': forecast_values.tolist(),
                'horizon_days': self.forecaster.forecast_horizon,
                'confidence_interval': {
                    'lower': lower_bound.tolist(),
                    'upper': upper_bound.tolist(),
                    'confidence_level': 0.80
                }
            },
            'alerts': {
                'total': len(alerts),
                'by_severity': self._count_alerts_by_severity(alerts),
                'details': [alert.to_dict() for alert in alerts]
            },
            'alert_summary': {
                'total_alerts': len(alerts),
                'by_severity': self._count_alerts_by_severity(alerts),
                'by_type': {'statistical_deviation': len(alerts)},
                'most_recent': alerts[0].to_dict() if alerts else None,
                'time_window_hours': 24
            }
        }
        
        # Add feedback summary if enabled (synthetic for demo)
        if self.enable_feedback_loop and include_feedback:
            response['feedback'] = {
                'summary': {
                    'total_signals': 0,
                    'avg_performance_score': 0.75,
                    'recent_trend': 'stable'
                },
                'strategy_weights': {
                    'price_sensitivity': 1.0,
                    'promotion_effectiveness': 1.0,
                    'sales_approach_impact': 1.0
                },
                'rl_weight_adjustments': {
                    'price_weight': 0.0,
                    'promotion_weight': 0.0,
                    'approach_weight': 0.0
                }
            }
        
        # Add trend analysis
        response['trend_analysis'] = self._analyze_trends(
            historical_data=historical_data,
            forecast_values=forecast_values
        )
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        response['processing_time_seconds'] = processing_time
        
        self.last_update_time = datetime.now()
        
        return response
