"""OpenAPI/Swagger documentation configuration"""

from flask import jsonify
import os


def get_openapi_spec():
    """
    Generate OpenAPI 3.0 specification for the API.
    
    Returns:
        OpenAPI specification dictionary
    """
    api_prefix = os.getenv('API_PREFIX', '/api/v1')
    
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "AI-Powered Enterprise Sales Strategist API",
            "version": "1.0.0",
            "description": "REST API for AI-powered sales strategy generation, market analysis, and business optimization",
            "contact": {
                "name": "API Support",
                "email": "support@example.com"
            }
        },
        "servers": [
            {
                "url": f"http://localhost:5000{api_prefix}",
                "description": "Development server"
            },
            {
                "url": f"https://api.example.com{api_prefix}",
                "description": "Production server"
            }
        ],
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key for authentication"
                },
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "JWT token for authentication"
                }
            },
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "message": {"type": "string"},
                        "status_code": {"type": "integer"}
                    }
                },
                "HealthCheck": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["healthy", "unhealthy"]},
                        "message": {"type": "string"},
                        "services": {"type": "object"}
                    }
                }
            }
        },
        "security": [
            {"ApiKeyAuth": []},
            {"BearerAuth": []}
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check endpoint",
                    "description": "Check the health status of the API and its services",
                    "tags": ["System"],
                    "security": [],
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthCheck"}
                                }
                            }
                        },
                        "503": {
                            "description": "Service is unhealthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/analyze_company": {
                "post": {
                    "summary": "Analyze company or product",
                    "description": "Analyze company or product text using BERT model to extract structured insights",
                    "tags": ["Enterprise Analyst"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["text"],
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "Company or product text to analyze",
                                            "maxLength": 200000
                                        },
                                        "source_type": {
                                            "type": "string",
                                            "enum": ["annual_report", "product_summary", "whitepaper"],
                                            "description": "Type of source document"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Analysis completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "product_category": {"type": "string"},
                                            "business_domain": {"type": "string"},
                                            "value_proposition": {"type": "string"},
                                            "key_features": {"type": "array", "items": {"type": "string"}},
                                            "confidence_scores": {"type": "object"},
                                            "processing_time_ms": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Error"}
                                }
                            }
                        }
                    }
                }
            },
            "/market_analysis": {
                "post": {
                    "summary": "Analyze market data",
                    "description": "Identify customer segments and market relationships using autoencoders, clustering, and GNN",
                    "tags": ["Market Decipherer"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["market_data"],
                                    "properties": {
                                        "market_data": {
                                            "type": "array",
                                            "items": {"type": "object"},
                                            "description": "Market data entities"
                                        },
                                        "entity_ids": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "auto_select_clusters": {"type": "boolean", "default": True},
                                        "similarity_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                                        "top_k_links": {"type": "integer", "minimum": 1, "maximum": 100}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Market analysis completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "clusters": {"type": "object"},
                                            "graph": {"type": "object"},
                                            "potential_clients": {"type": "array"},
                                            "processing_time_seconds": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/strategy": {
                "post": {
                    "summary": "Generate sales strategy",
                    "description": "Generate optimal pricing and sales strategy using RL agent and LLM",
                    "tags": ["Strategy Engine"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["market_state"],
                                    "properties": {
                                        "market_state": {
                                            "type": "object",
                                            "required": ["market_demand", "competitor_prices", "sales_volume", "conversion_rate", "inventory_level", "market_trend"],
                                            "properties": {
                                                "market_demand": {"type": "number", "minimum": 0, "maximum": 1},
                                                "competitor_prices": {"type": "array", "items": {"type": "number"}},
                                                "sales_volume": {"type": "number", "minimum": 0, "maximum": 1},
                                                "conversion_rate": {"type": "number", "minimum": 0, "maximum": 1},
                                                "inventory_level": {"type": "number", "minimum": 0, "maximum": 1},
                                                "market_trend": {"type": "number", "minimum": -1, "maximum": 1}
                                            }
                                        },
                                        "context": {"type": "object"},
                                        "include_explanation": {"type": "boolean", "default": True},
                                        "deterministic": {"type": "boolean", "default": True}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Strategy generated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "recommendations": {"type": "object"},
                                            "confidence_score": {"type": "number"},
                                            "explanation": {"type": "object"},
                                            "actionable_insights": {"type": "array"}
                                        }
                                    }
                                }
                            }
                        },
                        "429": {
                            "description": "Rate limit exceeded"
                        }
                    }
                }
            },
            "/performance": {
                "post": {
                    "summary": "Monitor performance",
                    "description": "Forecast sales trends and detect anomalies using LSTM",
                    "tags": ["Performance Governor"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["historical_data"],
                                    "properties": {
                                        "historical_data": {
                                            "type": "array",
                                            "items": {"type": "array", "items": {"type": "number"}}
                                        },
                                        "current_data": {
                                            "type": "array",
                                            "items": {"type": "array", "items": {"type": "number"}}
                                        },
                                        "strategy_context": {"type": "object"},
                                        "include_feedback": {"type": "boolean", "default": True}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Performance monitoring completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "forecast": {"type": "object"},
                                            "alerts": {"type": "object"},
                                            "feedback": {"type": "object"},
                                            "trend_analysis": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/business_optimizer": {
                "post": {
                    "summary": "Optimize business resources",
                    "description": "Optimize manufacturing and resource allocation using regression models",
                    "tags": ["Business Manager"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["product_portfolio"],
                                    "properties": {
                                        "product_portfolio": {
                                            "type": "array",
                                            "items": {"type": "object"}
                                        },
                                        "rl_strategy_outputs": {"type": "object"},
                                        "lstm_forecast_outputs": {"type": "object"},
                                        "constraints": {"type": "object"},
                                        "revenue_weight": {"type": "number", "minimum": 0, "maximum": 1},
                                        "cost_weight": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Optimization completed",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "production_priorities": {"type": "array"},
                                            "focus_products": {"type": "array"},
                                            "resource_allocation": {"type": "object"},
                                            "optimization_metrics": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/explain": {
                "post": {
                    "summary": "Explain model predictions",
                    "description": "Generate SHAP-based explanations for model predictions",
                    "tags": ["Model Transparency"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["model_type"],
                                    "properties": {
                                        "model_type": {
                                            "type": "string",
                                            "enum": ["rl", "lstm", "regression"]
                                        },
                                        "instance": {"type": "array"},
                                        "top_n": {"type": "integer", "default": 10},
                                        "include_visualizations": {"type": "boolean", "default": False},
                                        "explanation_type": {
                                            "type": "string",
                                            "enum": ["local", "global"],
                                            "default": "local"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Explanation generated",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "model_type": {"type": "string"},
                                            "explanation_type": {"type": "string"},
                                            "top_features": {"type": "array"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "tags": [
            {"name": "System", "description": "System health and information endpoints"},
            {"name": "Enterprise Analyst", "description": "Company and product analysis using BERT"},
            {"name": "Market Decipherer", "description": "Market segmentation and relationship analysis"},
            {"name": "Strategy Engine", "description": "Sales and pricing strategy generation"},
            {"name": "Performance Governor", "description": "Performance monitoring and forecasting"},
            {"name": "Business Manager", "description": "Resource optimization and business planning"},
            {"name": "Model Transparency", "description": "Model explainability and interpretability"}
        ]
    }
    
    return spec


def register_swagger_routes(app, api_prefix='/api/v1'):
    """
    Register Swagger UI and OpenAPI spec routes.
    
    Args:
        app: Flask application instance
        api_prefix: API prefix path
    """
    
    @app.route(f'{api_prefix}/openapi.json')
    def openapi_spec():
        """Return OpenAPI specification"""
        return jsonify(get_openapi_spec())
    
    @app.route(f'{api_prefix}/docs')
    def swagger_ui():
        """Render Swagger UI"""
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Documentation - Sales Strategist</title>
            <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css">
            <style>
                body {{ margin: 0; padding: 0; }}
            </style>
        </head>
        <body>
            <div id="swagger-ui"></div>
            <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
            <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js"></script>
            <script>
                window.onload = function() {{
                    const ui = SwaggerUIBundle({{
                        url: "{api_prefix}/openapi.json",
                        dom_id: '#swagger-ui',
                        deepLinking: true,
                        presets: [
                            SwaggerUIBundle.presets.apis,
                            SwaggerUIStandalonePreset
                        ],
                        plugins: [
                            SwaggerUIBundle.plugins.DownloadUrl
                        ],
                        layout: "StandaloneLayout"
                    }});
                    window.ui = ui;
                }};
            </script>
        </body>
        </html>
        '''
