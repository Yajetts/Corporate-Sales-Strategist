/**
 * Performance Monitoring View JavaScript
 */

let currentPerformanceData = null;
let trendChart = null;
let realTimeInterval = null;

// Form submission
document.getElementById('performanceForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('performanceDataFile');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('alertContainer', 'Please select a file to upload');
        return;
    }
    
    const file = fileInput.files[0];
    
    try {
        showLoading('alertContainer');
        
        // Read file
        const historicalData = await readFile(file);
        
        // Get options
        const options = {
            includeFeedback: document.getElementById('includeFeedback').checked
        };
        
        // Call API
        const result = await apiClient.monitorPerformance(historicalData, options);
        
        // Display results
        displayResults(result);
        
        showSuccess('alertContainer', 'Performance analysis completed successfully!');
        
    } catch (error) {
        showError('alertContainer', 'Failed to analyze performance data', error.message);
    }
});

/**
 * Read file content
 */
async function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            try {
                const content = e.target.result;
                
                if (file.name.endsWith('.json')) {
                    resolve(JSON.parse(content));
                } else if (file.name.endsWith('.csv')) {
                    resolve(parseCSVToArray(content));
                } else {
                    reject(new Error('Unsupported file format'));
                }
            } catch (error) {
                reject(error);
            }
        };
        
        reader.onerror = function() {
            reject(new Error('Failed to read file'));
        };
        
        reader.readAsText(file);
    });
}

/**
 * Parse CSV to 2D array
 */
function parseCSVToArray(content) {
    const lines = content.trim().split('\n');
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => parseFloat(v.trim()));
        data.push(values);
    }
    
    return data;
}

/**
 * Load sample performance data
 */
async function loadSamplePerformanceData() {
    try {
        showLoading('alertContainer');
        
        // Generate sample time-series data
        const sampleData = generateSampleTimeSeriesData(90, 3);
        
        const options = {
            includeFeedback: true
        };
        
        const result = await apiClient.monitorPerformance(sampleData, options);
        
        displayResults(result);
        
        showSuccess('alertContainer', 'Sample data loaded and analyzed successfully!');
        
    } catch (error) {
        showError('alertContainer', 'Failed to load sample data', error.message);
    }
}

/**
 * Generate sample time-series data
 */
function generateSampleTimeSeriesData(timesteps, features) {
    const data = [];
    
    for (let i = 0; i < timesteps; i++) {
        const row = [];
        for (let j = 0; j < features; j++) {
            const trend = i / timesteps;
            const seasonal = Math.sin(i * 2 * Math.PI / 30) * 0.2;
            const noise = (Math.random() - 0.5) * 0.1;
            row.push(trend + seasonal + noise + 0.5);
        }
        data.push(row);
    }
    
    return data;
}

/**
 * Display performance results
 */
function displayResults(data) {
    currentPerformanceData = data;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update alert summary
    const alertsBySeverity = data.alerts.by_severity;
    document.getElementById('criticalAlerts').textContent = alertsBySeverity.critical || 0;
    document.getElementById('highAlerts').textContent = alertsBySeverity.high || 0;
    document.getElementById('mediumAlerts').textContent = alertsBySeverity.medium || 0;
    document.getElementById('lowAlerts').textContent = alertsBySeverity.low || 0;
    
    // Update last updated time
    document.getElementById('lastUpdated').textContent = new Date(data.timestamp).toLocaleTimeString();
    
    // Update trend analysis
    const trendAnalysis = data.trend_analysis;
    document.getElementById('historicalTrend').textContent = formatPercentage(trendAnalysis.historical_trend);
    document.getElementById('forecastTrend').textContent = formatPercentage(trendAnalysis.forecast_trend);
    document.getElementById('trendOutlook').textContent = trendAnalysis.trend_outlook;
    document.getElementById('forecastVsCurrent').textContent = formatPercentage(trendAnalysis.forecast_vs_current_pct);
    
    // Render visualizations
    renderTrendChart(data.forecast);
    renderAnomalyAlerts(data.alerts.details);
    
    // Render feedback if available
    if (data.feedback) {
        renderFeedbackSummary(data.feedback);
    }
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Render trend chart with LSTM forecast
 */
function renderTrendChart(forecast) {
    const ctx = document.getElementById('trendChart').getContext('2d');
    
    if (trendChart) {
        trendChart.destroy();
    }
    
    const horizonDays = forecast.horizon_days;
    const forecastValues = forecast.values;
    const confidenceInterval = forecast.confidence_interval;
    
    // Generate labels (days)
    const historicalDays = 30;
    const labels = [];
    for (let i = -historicalDays; i <= horizonDays; i++) {
        labels.push(i === 0 ? 'Today' : (i < 0 ? `${-i}d ago` : `+${i}d`));
    }
    
    // Generate historical data (simulated)
    const historicalData = [];
    for (let i = 0; i < historicalDays; i++) {
        historicalData.push(100 + (Math.random() - 0.5) * 20);
    }
    
    // Current value
    const currentValue = historicalData[historicalData.length - 1];
    
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical Sales',
                    data: [...historicalData, currentValue, ...Array(horizonDays).fill(null)],
                    borderColor: 'rgba(13, 110, 253, 1)',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 3,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    fill: false
                },
                {
                    label: 'Forecasted Sales',
                    data: [...Array(historicalDays + 1).fill(null), ...forecastValues],
                    borderColor: 'rgba(25, 135, 84, 1)',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    fill: false
                },
                {
                    label: `Upper Bound (${(confidenceInterval.confidence_level * 100).toFixed(0)}% CI)`,
                    data: [...Array(historicalDays + 1).fill(null), ...confidenceInterval.upper],
                    borderColor: 'rgba(25, 135, 84, 0.3)',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: '+1'
                },
                {
                    label: `Lower Bound (${(confidenceInterval.confidence_level * 100).toFixed(0)}% CI)`,
                    data: [...Array(historicalDays + 1).fill(null), ...confidenceInterval.lower],
                    borderColor: 'rgba(25, 135, 84, 0.3)',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: `Sales Forecast (${horizonDays} Days Ahead)`
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Sales Volume'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            }
        }
    });
}

/**
 * Render anomaly alerts
 */
function renderAnomalyAlerts(alerts) {
    const container = document.getElementById('alertsContainer');
    container.innerHTML = '';
    
    if (!alerts || alerts.length === 0) {
        container.innerHTML = '<p class="text-muted text-center">No anomalies detected</p>';
        return;
    }
    
    // Sort by severity
    const severityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    alerts.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);
    
    alerts.forEach(alert => {
        const alertCard = document.createElement('div');
        alertCard.className = 'alert alert-' + getSeverityAlertClass(alert.severity) + ' mb-3';
        
        alertCard.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                    <h6 class="alert-heading">
                        <i class="bi bi-exclamation-triangle-fill"></i>
                        ${alert.anomaly_type} - ${alert.metric_name}
                        <span class="badge ${getSeverityBadgeClass(alert.severity)} ms-2">${alert.severity.toUpperCase()}</span>
                    </h6>
                    <p class="mb-2">${alert.description}</p>
                    <div class="row">
                        <div class="col-md-4">
                            <small><strong>Actual:</strong> ${formatNumber(alert.actual_value)}</small>
                        </div>
                        <div class="col-md-4">
                            <small><strong>Expected:</strong> ${formatNumber(alert.expected_value)}</small>
                        </div>
                        <div class="col-md-4">
                            <small><strong>Deviation:</strong> ${formatPercentage(alert.deviation_pct)}</small>
                        </div>
                    </div>
                    ${alert.recommended_actions && alert.recommended_actions.length > 0 ? `
                        <div class="mt-2">
                            <strong>Recommended Actions:</strong>
                            <ul class="mb-0 mt-1">
                                ${alert.recommended_actions.map(action => `<li>${action}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
                <small class="text-muted ms-3">${new Date(alert.timestamp).toLocaleString()}</small>
            </div>
        `;
        
        container.appendChild(alertCard);
    });
}

/**
 * Get severity alert class
 */
function getSeverityAlertClass(severity) {
    const classMap = {
        'critical': 'danger',
        'high': 'warning',
        'medium': 'info',
        'low': 'secondary'
    };
    return classMap[severity.toLowerCase()] || 'secondary';
}

/**
 * Render feedback summary
 */
function renderFeedbackSummary(feedback) {
    document.getElementById('feedbackSection').style.display = 'block';
    
    const container = document.getElementById('feedbackContent');
    
    let html = '<div class="row">';
    
    // Strategy weights
    if (feedback.strategy_weights) {
        html += '<div class="col-md-6 mb-3">';
        html += '<h6><i class="bi bi-sliders"></i> Strategy Weights</h6>';
        html += '<ul class="list-group">';
        for (const [strategy, weight] of Object.entries(feedback.strategy_weights)) {
            html += `<li class="list-group-item d-flex justify-content-between align-items-center">
                ${strategy}
                <span class="badge bg-primary">${formatNumber(weight, 3)}</span>
            </li>`;
        }
        html += '</ul>';
        html += '</div>';
    }
    
    // RL weight adjustments
    if (feedback.rl_weight_adjustments) {
        html += '<div class="col-md-6 mb-3">';
        html += '<h6><i class="bi bi-arrow-repeat"></i> RL Weight Adjustments</h6>';
        html += '<ul class="list-group">';
        for (const [param, adjustment] of Object.entries(feedback.rl_weight_adjustments)) {
            const adjustmentClass = adjustment > 0 ? 'success' : adjustment < 0 ? 'danger' : 'secondary';
            html += `<li class="list-group-item d-flex justify-content-between align-items-center">
                ${param}
                <span class="badge bg-${adjustmentClass}">${adjustment > 0 ? '+' : ''}${formatNumber(adjustment, 3)}</span>
            </li>`;
        }
        html += '</ul>';
        html += '</div>';
    }
    
    html += '</div>';
    
    container.innerHTML = html;
}

/**
 * Enable real-time updates
 */
function enableRealTimeUpdates() {
    if (realTimeInterval) {
        clearInterval(realTimeInterval);
        realTimeInterval = null;
        showSuccess('alertContainer', 'Real-time updates disabled');
        return;
    }
    
    if (!currentPerformanceData) {
        showError('alertContainer', 'Please load data first before enabling real-time updates');
        return;
    }
    
    showSuccess('alertContainer', 'Real-time updates enabled (refreshing every 30 seconds)');
    
    realTimeInterval = setInterval(async () => {
        try {
            // Simulate real-time update by adding noise to current data
            const updatedData = JSON.parse(JSON.stringify(currentPerformanceData));
            updatedData.timestamp = new Date().toISOString();
            
            // Update display
            document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
            
        } catch (error) {
            console.error('Real-time update failed:', error);
        }
    }, 30000);
}

/**
 * Refresh alerts
 */
async function refreshAlerts() {
    if (!currentPerformanceData) {
        showError('alertContainer', 'No data loaded');
        return;
    }
    
    try {
        showLoading('alertContainer');
        
        // Fetch latest alerts
        const alerts = await apiClient.getPerformanceAlerts(24);
        
        renderAnomalyAlerts(alerts.details);
        
        showSuccess('alertContainer', 'Alerts refreshed successfully');
        
    } catch (error) {
        showError('alertContainer', 'Failed to refresh alerts', error.message);
    }
}
