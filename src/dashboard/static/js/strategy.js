/**
 * Strategy Generation View JavaScript
 */

let currentStrategyData = null;
let revenueForecastChart = null;

// Update slider values
const sliders = ['marketDemand', 'salesVolume', 'conversionRate', 'inventoryLevel', 'marketTrend'];
sliders.forEach(sliderId => {
    const slider = document.getElementById(sliderId);
    const valueDisplay = document.getElementById(sliderId + 'Value');
    
    slider.addEventListener('input', function(e) {
        valueDisplay.textContent = parseFloat(e.target.value).toFixed(2);
    });
});

// Form submission
document.getElementById('strategyForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    try {
        showLoading('alertContainer');
        
        // Get market state
        const marketState = getMarketState();
        
        // Get options
        const options = {
            includeExplanation: document.getElementById('includeExplanation').checked
        };
        
        // Call API
        const result = await apiClient.generateStrategy(marketState, options);
        
        // Display results
        displayResults(result);
        
        showSuccess('alertContainer', 'Strategy generated successfully!');
        
    } catch (error) {
        showError('alertContainer', 'Failed to generate strategy', error.message);
    }
});

/**
 * Get market state from form
 */
function getMarketState() {
    const competitorPricesStr = document.getElementById('competitorPrices').value;
    const competitorPrices = competitorPricesStr.split(',').map(p => parseFloat(p.trim()));
    
    return {
        market_demand: parseFloat(document.getElementById('marketDemand').value),
        competitor_prices: competitorPrices,
        sales_volume: parseFloat(document.getElementById('salesVolume').value),
        conversion_rate: parseFloat(document.getElementById('conversionRate').value),
        inventory_level: parseFloat(document.getElementById('inventoryLevel').value),
        market_trend: parseFloat(document.getElementById('marketTrend').value)
    };
}

/**
 * Load preset scenario
 */
async function loadPresetScenario() {
    // Set preset values
    document.getElementById('marketDemand').value = 0.75;
    document.getElementById('marketDemandValue').textContent = '0.75';
    
    document.getElementById('salesVolume').value = 0.65;
    document.getElementById('salesVolumeValue').textContent = '0.65';
    
    document.getElementById('conversionRate').value = 0.55;
    document.getElementById('conversionRateValue').textContent = '0.55';
    
    document.getElementById('inventoryLevel').value = 0.70;
    document.getElementById('inventoryLevelValue').textContent = '0.70';
    
    document.getElementById('marketTrend').value = 0.15;
    document.getElementById('marketTrendValue').textContent = '0.15';
    
    document.getElementById('competitorPrices').value = '0.45, 0.55, 0.65';
    
    showSuccess('alertContainer', 'Preset scenario loaded. Click "Generate Strategy" to analyze.');
}

/**
 * Display strategy results
 */
function displayResults(data) {
    currentStrategyData = data;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update recommendations
    const recommendations = data.recommendations;
    document.getElementById('priceAdjustment').textContent = formatPercentage(recommendations.price_adjustment_pct);
    document.getElementById('salesApproach').textContent = recommendations.sales_approach.toUpperCase();
    document.getElementById('promotionIntensity').textContent = formatPercentage(recommendations.promotion_intensity);
    
    // Update confidence
    const confidenceScore = data.confidence_score;
    const confidenceClass = getConfidenceClass(confidenceScore);
    const confidenceText = getConfidenceText(confidenceScore);
    
    document.getElementById('confidenceBar').className = 'confidence-fill ' + confidenceClass;
    document.getElementById('confidenceBar').style.width = (confidenceScore * 100) + '%';
    document.getElementById('confidenceBadge').textContent = confidenceText + ' Confidence';
    document.getElementById('confidenceBadge').className = 'badge badge-lg ' + 
        (confidenceScore >= 0.8 ? 'bg-success' : confidenceScore >= 0.5 ? 'bg-warning' : 'bg-danger');
    document.getElementById('confidenceScore').textContent = formatPercentage(confidenceScore);
    
    // Update explanation if available
    if (data.explanation) {
        document.getElementById('explanationSection').style.display = 'block';
        document.getElementById('explanationSummary').textContent = data.explanation.summary || 'N/A';
        document.getElementById('explanationRationale').textContent = data.explanation.rationale || 'N/A';
        document.getElementById('explanationOutcomes').textContent = data.explanation.expected_outcomes || 'N/A';
        document.getElementById('explanationRisks').textContent = data.explanation.risks || 'N/A';
    } else {
        document.getElementById('explanationSection').style.display = 'none';
    }
    
    // Render charts
    renderRevenueForecastChart(data);
    renderPerformanceHeatmap(data);
    
    // Render actionable insights
    renderActionableInsights(data.actionable_insights || []);
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Render revenue forecast chart
 */
function renderRevenueForecastChart(data) {
    const ctx = document.getElementById('revenueForecastChart').getContext('2d');
    
    if (revenueForecastChart) {
        revenueForecastChart.destroy();
    }
    
    // Generate forecast data (simulated)
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const baseRevenue = 100000;
    const priceAdjustment = data.recommendations.price_adjustment_pct;
    
    // Historical data (6 months)
    const historicalData = months.slice(0, 6).map((_, i) => {
        return baseRevenue * (1 + (Math.random() - 0.5) * 0.2);
    });
    
    // Forecast data (6 months)
    const forecastData = months.slice(6).map((_, i) => {
        return baseRevenue * (1 + priceAdjustment) * (1 + (Math.random() - 0.5) * 0.15);
    });
    
    // Confidence intervals
    const upperBound = forecastData.map(v => v * 1.15);
    const lowerBound = forecastData.map(v => v * 0.85);
    
    revenueForecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Historical Revenue',
                    data: [...historicalData, ...Array(6).fill(null)],
                    borderColor: 'rgba(13, 110, 253, 1)',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7
                },
                {
                    label: 'Forecasted Revenue',
                    data: [...Array(6).fill(null), ...forecastData],
                    borderColor: 'rgba(25, 135, 84, 1)',
                    backgroundColor: 'rgba(25, 135, 84, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    pointRadius: 5,
                    pointHoverRadius: 7
                },
                {
                    label: 'Upper Bound (85% CI)',
                    data: [...Array(6).fill(null), ...upperBound],
                    borderColor: 'rgba(25, 135, 84, 0.3)',
                    backgroundColor: 'rgba(25, 135, 84, 0.05)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: '+1'
                },
                {
                    label: 'Lower Bound (85% CI)',
                    data: [...Array(6).fill(null), ...lowerBound],
                    borderColor: 'rgba(25, 135, 84, 0.3)',
                    backgroundColor: 'rgba(25, 135, 84, 0.05)',
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
                    text: 'Revenue Forecast with Confidence Intervals'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += '$' + formatNumber(context.parsed.y, 0);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + (value / 1000).toFixed(0) + 'K';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Render performance heatmap
 */
function renderPerformanceHeatmap(data) {
    const container = document.getElementById('performanceHeatmap');
    
    // Generate heatmap data for different strategy combinations
    const priceAdjustments = [-0.2, -0.1, 0, 0.1, 0.2];
    const promotionLevels = [0.2, 0.4, 0.6, 0.8, 1.0];
    
    const zData = [];
    for (let i = 0; i < promotionLevels.length; i++) {
        const row = [];
        for (let j = 0; j < priceAdjustments.length; j++) {
            // Simulate performance score
            const baseScore = 0.5;
            const priceEffect = -priceAdjustments[j] * 0.3;
            const promoEffect = promotionLevels[i] * 0.4;
            const randomNoise = (Math.random() - 0.5) * 0.1;
            row.push(Math.max(0, Math.min(1, baseScore + priceEffect + promoEffect + randomNoise)));
        }
        zData.push(row);
    }
    
    const heatmapData = [{
        z: zData,
        x: priceAdjustments.map(p => formatPercentage(p)),
        y: promotionLevels.map(p => formatPercentage(p)),
        type: 'heatmap',
        colorscale: [
            [0, 'rgb(220, 53, 69)'],
            [0.5, 'rgb(255, 193, 7)'],
            [1, 'rgb(25, 135, 84)']
        ],
        hoverongaps: false,
        hovertemplate: 'Price: %{x}<br>Promotion: %{y}<br>Performance: %{z:.2f}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'Strategy Performance Matrix',
            font: { size: 18 }
        },
        xaxis: {
            title: 'Price Adjustment',
            side: 'bottom'
        },
        yaxis: {
            title: 'Promotion Intensity'
        },
        margin: { t: 50, b: 80, l: 80, r: 50 }
    };
    
    Plotly.newPlot('performanceHeatmap', heatmapData, layout, { responsive: true });
    
    // Add marker for current strategy
    const currentPrice = data.recommendations.price_adjustment_pct;
    const currentPromo = data.recommendations.promotion_intensity;
    
    const markerData = {
        x: [formatPercentage(currentPrice)],
        y: [formatPercentage(currentPromo)],
        mode: 'markers',
        marker: {
            size: 15,
            color: 'white',
            symbol: 'star',
            line: {
                color: 'black',
                width: 2
            }
        },
        name: 'Current Strategy',
        hovertemplate: 'Current Strategy<extra></extra>'
    };
    
    Plotly.addTraces('performanceHeatmap', markerData);
}

/**
 * Render actionable insights
 */
function renderActionableInsights(insights) {
    const list = document.getElementById('actionableInsightsList');
    list.innerHTML = '';
    
    if (!insights || insights.length === 0) {
        list.innerHTML = '<li class="list-group-item text-muted">No actionable insights available</li>';
        return;
    }
    
    insights.forEach((insight, index) => {
        const item = document.createElement('li');
        item.className = 'list-group-item';
        item.innerHTML = `
            <div class="d-flex align-items-start">
                <span class="badge bg-primary me-3">${index + 1}</span>
                <span>${insight}</span>
            </div>
        `;
        list.appendChild(item);
    });
}
