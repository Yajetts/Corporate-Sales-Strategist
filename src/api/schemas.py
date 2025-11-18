"""Request and response schemas for API validation"""

from marshmallow import Schema, fields, validate, ValidationError, validates


class AnalyzeCompanyRequestSchema(Schema):
    """Schema for /api/v1/analyze_company request"""
    
    text = fields.Str(
        required=True,
        validate=validate.Length(min=1, max=200000),
        error_messages={
            'required': 'Text field is required',
            'invalid': 'Text must be a string'
        }
    )
    
    source_type = fields.Str(
        required=False,
        allow_none=True,
        validate=validate.OneOf(
            ['annual_report', 'product_summary', 'whitepaper'],
            error='Invalid source_type. Must be one of: annual_report, product_summary, whitepaper'
        )
    )



class AnalyzeCompanyResponseSchema(Schema):
    """Schema for /api/v1/analyze_company response"""
    
    product_category = fields.Str(required=True)
    business_domain = fields.Str(required=True)
    value_proposition = fields.Str(required=True)
    key_features = fields.List(fields.Str(), required=True)
    confidence_scores = fields.Dict(
        keys=fields.Str(),
        values=fields.Float(),
        required=True
    )
    processing_time_ms = fields.Int(required=True)
    source_type = fields.Str(allow_none=True)


class ErrorResponseSchema(Schema):
    """Schema for error responses"""
    
    error = fields.Str(required=True)
    message = fields.Str(required=True)
    status_code = fields.Int(required=True)


class MarketAnalysisRequestSchema(Schema):
    """Schema for /api/v1/market_analysis request"""
    
    market_data = fields.List(
        fields.Dict(),
        required=True,
        validate=validate.Length(min=1, max=10000),
        error_messages={
            'required': 'market_data field is required',
            'invalid': 'market_data must be a list of dictionaries'
        }
    )
    
    entity_ids = fields.List(
        fields.Str(),
        required=False,
        allow_none=True
    )
    
    auto_select_clusters = fields.Bool(
        required=False,
        load_default=True
    )
    
    similarity_threshold = fields.Float(
        required=False,
        load_default=0.7,
        validate=validate.Range(min=0.0, max=1.0)
    )
    
    top_k_links = fields.Int(
        required=False,
        load_default=20,
        validate=validate.Range(min=1, max=100)
    )



class MarketAnalysisResponseSchema(Schema):
    """Schema for /api/v1/market_analysis response"""
    
    clusters = fields.Dict(required=True)
    graph = fields.Dict(required=True)
    potential_clients = fields.List(fields.Dict(), required=True)
    latent_dimensions = fields.Int(required=True)
    processing_time_seconds = fields.Float(required=True)
    num_entities = fields.Int(required=True)
    task_id = fields.Str(required=False, allow_none=True)


class StrategyRequestSchema(Schema):
    """Schema for /api/v1/strategy request"""
    
    market_state = fields.Dict(
        required=True,
        error_messages={
            'required': 'market_state field is required',
            'invalid': 'market_state must be a dictionary'
        }
    )
    
    context = fields.Dict(
        required=False,
        allow_none=True
    )
    
    include_explanation = fields.Bool(
        required=False,
        load_default=True
    )
    
    deterministic = fields.Bool(
        required=False,
        load_default=True
    )



class StrategyResponseSchema(Schema):
    """Schema for /api/v1/strategy response"""
    
    timestamp = fields.Str(required=True)
    market_state = fields.Dict(required=True)
    recommendations = fields.Dict(required=True)
    confidence_score = fields.Float(required=True)
    confidence_level = fields.Str(required=True)
    explanation = fields.Dict(required=False, allow_none=True)
    actionable_insights = fields.List(fields.Str(), required=True)
    task_id = fields.Str(required=False, allow_none=True)



class PerformanceMonitoringRequestSchema(Schema):
    """Schema for /api/v1/performance request"""
    
    historical_data = fields.List(
        fields.List(fields.Float()),
        required=True,
        validate=validate.Length(min=1),
        error_messages={
            'required': 'historical_data field is required',
            'invalid': 'historical_data must be a 2D array of floats'
        }
    )
    
    current_data = fields.List(
        fields.List(fields.Float()),
        required=False,
        allow_none=True
    )
    
    strategy_context = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=False,
        allow_none=True
    )
    
    include_feedback = fields.Bool(
        required=False,
        load_default=True
    )


class PerformanceMonitoringResponseSchema(Schema):
    """Schema for /api/v1/performance response"""
    
    timestamp = fields.Str(required=True)
    
    forecast = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=True
    )
    
    alerts = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=True
    )
    
    alert_summary = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=True
    )
    
    feedback = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=False,
        allow_none=True
    )
    
    trend_analysis = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=True
    )
    
    processing_time_seconds = fields.Float(required=True)


class BusinessOptimizerRequestSchema(Schema):
    """Schema for /api/v1/business_optimizer request"""
    
    product_portfolio = fields.List(
        fields.Dict(),
        required=True,
        validate=validate.Length(min=1, max=100),
        error_messages={
            'required': 'product_portfolio field is required',
            'invalid': 'product_portfolio must be a list of dictionaries'
        }
    )
    
    rl_strategy_outputs = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=False,
        allow_none=True
    )
    
    lstm_forecast_outputs = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=False,
        allow_none=True
    )
    
    constraints = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=False,
        allow_none=True
    )
    
    revenue_weight = fields.Float(
        required=False,
        load_default=0.7,
        validate=validate.Range(min=0.0, max=1.0)
    )
    
    cost_weight = fields.Float(
        required=False,
        load_default=0.3,
        validate=validate.Range(min=0.0, max=1.0)
    )



class BusinessOptimizerResponseSchema(Schema):
    """Schema for /api/v1/business_optimizer response"""
    
    timestamp = fields.Str(required=True)
    production_priorities = fields.List(fields.Dict(), required=True)
    recommendations = fields.List(fields.Dict(), required=False)  # Alias for backward compatibility
    focus_products = fields.List(fields.Dict(), required=True)
    resource_allocation = fields.Dict(required=True)
    optimization_metrics = fields.Dict(required=True)
    constraints_applied = fields.Bool(required=True)
    optimization_success = fields.Bool(required=True)
    processing_time_seconds = fields.Float(required=True)
    num_products = fields.Int(required=True)
