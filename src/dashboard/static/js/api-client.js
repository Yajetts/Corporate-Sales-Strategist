/**
 * API Client for Sales Strategist Dashboard
 * Handles all REST API communication with error handling and loading states
 */

class APIClient {
    constructor(baseURL = '/api/v1') {
        this.baseURL = baseURL;
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }

    /**
     * Make a GET request
     */
    async get(endpoint, options = {}) {
        const cacheKey = `GET:${endpoint}`;
        
        // Check cache if enabled
        if (options.useCache !== false && this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                console.log(`Cache hit for ${endpoint}`);
                return cached.data;
            }
        }

        const url = this.baseURL + endpoint;
        
        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            const data = await this.handleResponse(response);
            
            // Cache successful responses
            if (options.useCache !== false) {
                this.cache.set(cacheKey, {
                    data: data,
                    timestamp: Date.now()
                });
            }
            
            return data;
        } catch (error) {
            console.error(`GET ${endpoint} failed:`, error);
            throw error;
        }
    }

    /**
     * Make a POST request
     */
    async post(endpoint, data, options = {}) {
        const url = this.baseURL + endpoint;
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                body: JSON.stringify(data)
            });

            return await this.handleResponse(response);
        } catch (error) {
            console.error(`POST ${endpoint} failed:`, error);
            throw error;
        }
    }

    /**
     * Handle API response
     */
    async handleResponse(response) {
        const contentType = response.headers.get('content-type');
        
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Invalid response format: expected JSON');
        }

        const data = await response.json();

        if (!response.ok) {
            const error = new Error(data.message || 'API request failed');
            error.status = response.status;
            error.data = data;
            throw error;
        }

        return data;
    }

    /**
     * Clear cache
     */
    clearCache(endpoint = null) {
        if (endpoint) {
            const cacheKey = `GET:${endpoint}`;
            this.cache.delete(cacheKey);
        } else {
            this.cache.clear();
        }
    }

    /**
     * Analyze company
     */
    async analyzeCompany(text, sourceType = null) {
        return await this.post('/analyze_company', {
            text: text,
            source_type: sourceType
        });
    }

    /**
     * Market analysis
     */
    async marketAnalysis(marketData, options = {}) {
        return await this.post('/market_analysis', {
            market_data: marketData,
            entity_ids: options.entityIds,
            auto_select_clusters: options.autoSelectClusters !== false,
            similarity_threshold: options.similarityThreshold || 0.7,
            top_k_links: options.topKLinks || 20
        });
    }

    /**
     * Generate strategy
     */
    async generateStrategy(marketState, options = {}) {
        return await this.post('/strategy', {
            market_state: marketState,
            context: options.context,
            include_explanation: options.includeExplanation !== false,
            deterministic: options.deterministic !== false
        });
    }

    /**
     * Compare strategies
     */
    async compareStrategies(scenarios) {
        return await this.post('/strategy/compare', {
            scenarios: scenarios
        });
    }

    /**
     * Monitor performance
     */
    async monitorPerformance(historicalData, options = {}) {
        return await this.post('/performance', {
            historical_data: historicalData,
            current_data: options.currentData,
            strategy_context: options.strategyContext,
            include_feedback: options.includeFeedback !== false
        });
    }

    /**
     * Get performance alerts
     */
    async getPerformanceAlerts(timeWindowHours = 24) {
        return await this.get(`/performance/alerts?time_window_hours=${timeWindowHours}`);
    }

    /**
     * Business optimization
     */
    async businessOptimization(products, options = {}) {
        return await this.post('/business_optimizer', {
            product_portfolio: products,
            constraints: options.constraints,
            rl_strategy_outputs: options.rl_strategy_outputs,
            lstm_forecast_outputs: options.lstm_forecast_outputs
        });
    }

    /**
     * Get model explanation
     */
    async getExplanation(modelType, prediction, features) {
        return await this.post('/explain', {
            model_type: modelType,
            prediction: prediction,
            features: features
        });
    }

    /**
     * Health check
     */
    async healthCheck() {
        return await this.get('/health', { useCache: false });
    }

    /**
     * Get model info
     */
    async getModelInfo() {
        return await this.get('/model_info');
    }
}

// Create global instance
const apiClient = new APIClient();

/**
 * UI Helper Functions
 */

/**
 * Show loading spinner
 */
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="loading-overlay">
                <div class="spinner-container">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Loading...</p>
                </div>
            </div>
        `;
    }
}

/**
 * Hide loading spinner
 */
function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        const overlay = element.querySelector('.loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }
}

/**
 * Show error message
 */
function showError(elementId, message, details = null) {
    const element = document.getElementById(elementId);
    if (element) {
        let html = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <h5 class="alert-heading"><i class="bi bi-exclamation-triangle"></i> Error</h5>
                <p>${message}</p>
        `;
        
        if (details) {
            html += `<hr><small>${details}</small>`;
        }
        
        html += `
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        element.innerHTML = html;
    }
}

/**
 * Show success message
 */
function showSuccess(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                <i class="bi bi-check-circle"></i> ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
    }
}

/**
 * Format number with commas
 */
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return 'N/A';
    return num.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/**
 * Format percentage
 */
function formatPercentage(value, decimals = 1) {
    if (value === null || value === undefined) return 'N/A';
    return (value * 100).toFixed(decimals) + '%';
}

/**
 * Get confidence level class
 */
function getConfidenceClass(score) {
    if (score >= 0.8) return 'confidence-high';
    if (score >= 0.5) return 'confidence-medium';
    return 'confidence-low';
}

/**
 * Get confidence level text
 */
function getConfidenceText(score) {
    if (score >= 0.8) return 'High';
    if (score >= 0.5) return 'Medium';
    return 'Low';
}

/**
 * Get severity badge class
 */
function getSeverityBadgeClass(severity) {
    const severityMap = {
        'critical': 'danger',
        'high': 'warning',
        'medium': 'info',
        'low': 'secondary'
    };
    return 'bg-' + (severityMap[severity.toLowerCase()] || 'secondary');
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Download data as JSON
 */
function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Download chart as PNG
 */
function downloadChartAsPNG(chartId, filename) {
    const canvas = document.getElementById(chartId);
    if (canvas && canvas.tagName === 'CANVAS') {
        const url = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}
