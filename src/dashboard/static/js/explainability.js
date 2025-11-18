/**
 * Explainability View JavaScript
 */

let currentExplanationData = null;
let featureImportanceChart = null;

// Form submission
document.getElementById('explainabilityForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    try {
        showLoading('alertContainer');
        
        // Get form data
        const modelType = document.getElementById('modelType').value;
        const explanationType = document.getElementById('explanationType').value;
        const predictionDataStr = document.getElementById('predictionData').value;
        
        if (!predictionDataStr.trim()) {
            showError('alertContainer', 'Please enter prediction data');
            return;
        }
        
        // Parse prediction data
        let predictionData;
        try {
            predictionData = JSON.parse(predictionDataStr);
        } catch (error) {
            showError('alertContainer', 'Invalid JSON format', error.message);
            return;
        }
        
        // Call API
        const result = await apiClient.getExplanation(modelType, predictionData, predictionData);
        
        // Display results
        displayResults(result, explanationType);
        
        showSuccess('alertContainer', 'Explanation generated successfully!');
        
    } catch (error) {
        showError('alertContainer', 'Failed to generate explanation', error.message);
    }
});

/**
 * Load sample explanation
 */
async function loadSampleExplanation() {
    try {
        showLoading('alertContainer');
        
        // Generate sample SHAP data
        const sampleData = generateSampleSHAPData();
        
        displayResults(sampleData, 'local');
        
        showSuccess('alertContainer', 'Sample explanation loaded successfully!');
        
    } catch (error) {
        showError('alertContainer', 'Failed to load sample explanation', error.message);
    }
}

/**
 * Generate sample SHAP data
 */
function generateSampleSHAPData() {
    const features = [
        'market_demand', 'competitor_price', 'sales_volume', 'conversion_rate',
        'inventory_level', 'market_trend', 'customer_satisfaction', 'product_quality',
        'brand_awareness', 'seasonal_factor'
    ];
    
    const shapValues = {};
    const featureValues = {};
    
    features.forEach(feature => {
        shapValues[feature] = (Math.random() - 0.5) * 2;
        featureValues[feature] = Math.random();
    });
    
    return {
        shap_values: shapValues,
        feature_values: featureValues,
        base_value: 0.5,
        prediction: 0.65,
        top_features: features.slice(0, 10)
    };
}

/**
 * Display explanation results
 */
function displayResults(data, explanationType) {
    currentExplanationData = data;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Render visualizations
    renderFeatureImportanceChart(data);
    renderSHAPBarChart(data);
    renderSHAPForcePlot(data);
    renderSHAPWaterfallChart(data);
    renderInfluentialFeaturesTable(data);
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Render feature importance chart
 */
function renderFeatureImportanceChart(data) {
    const ctx = document.getElementById('featureImportanceChart').getContext('2d');
    
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }
    
    // Get absolute SHAP values for importance
    const shapValues = data.shap_values;
    const features = Object.keys(shapValues);
    const importance = features.map(f => Math.abs(shapValues[f]));
    
    // Sort by importance
    const sorted = features.map((f, i) => ({ feature: f, importance: importance[i] }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 10);
    
    featureImportanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sorted.map(s => s.feature),
            datasets: [{
                label: 'Feature Importance',
                data: sorted.map(s => s.importance),
                backgroundColor: 'rgba(13, 110, 253, 0.7)',
                borderColor: 'rgba(13, 110, 253, 1)',
                borderWidth: 2
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Top 10 Most Important Features'
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Absolute SHAP Value'
                    }
                }
            }
        }
    });
}

/**
 * Render SHAP bar chart with Plotly
 */
function renderSHAPBarChart(data) {
    const shapValues = data.shap_values;
    const features = Object.keys(shapValues);
    const values = features.map(f => shapValues[f]);
    
    // Sort by absolute value
    const sorted = features.map((f, i) => ({ feature: f, value: values[i] }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 15);
    
    const barData = [{
        x: sorted.map(s => s.value),
        y: sorted.map(s => s.feature),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map(s => s.value > 0 ? 'rgba(25, 135, 84, 0.7)' : 'rgba(220, 53, 69, 0.7)'),
            line: {
                color: sorted.map(s => s.value > 0 ? 'rgba(25, 135, 84, 1)' : 'rgba(220, 53, 69, 1)'),
                width: 2
            }
        },
        hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'SHAP Values by Feature',
            font: { size: 18 }
        },
        xaxis: {
            title: 'SHAP Value',
            zeroline: true,
            zerolinewidth: 2,
            zerolinecolor: 'black'
        },
        yaxis: {
            title: 'Feature'
        },
        margin: { t: 50, b: 50, l: 150, r: 50 },
        showlegend: false
    };
    
    Plotly.newPlot('shapBarChart', barData, layout, { responsive: true });
}

/**
 * Render SHAP force plot
 */
function renderSHAPForcePlot(data) {
    const container = document.getElementById('shapForcePlot');
    
    const baseValue = data.base_value || 0.5;
    const prediction = data.prediction || 0.65;
    const shapValues = data.shap_values;
    
    // Sort features by absolute SHAP value
    const features = Object.keys(shapValues);
    const sorted = features.map(f => ({ feature: f, value: shapValues[f] }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 10);
    
    // Create force plot visualization
    let html = '<div class="force-plot-container">';
    html += `<div class="text-center mb-3">`;
    html += `<span class="badge bg-secondary me-2">Base Value: ${formatNumber(baseValue, 3)}</span>`;
    html += `<span class="badge bg-primary">Prediction: ${formatNumber(prediction, 3)}</span>`;
    html += `</div>`;
    
    html += '<div class="force-plot-bars">';
    
    let cumulative = baseValue;
    sorted.forEach(item => {
        const width = Math.abs(item.value) * 100;
        const color = item.value > 0 ? '#198754' : '#dc3545';
        const direction = item.value > 0 ? 'right' : 'left';
        
        html += `<div class="force-bar" style="width: ${width}%; background-color: ${color};" 
                      title="${item.feature}: ${formatNumber(item.value, 4)}">
            <span class="force-label">${item.feature}</span>
        </div>`;
        
        cumulative += item.value;
    });
    
    html += '</div>';
    html += '</div>';
    
    // Add CSS for force plot
    html += `<style>
        .force-plot-container {
            padding: 20px;
        }
        .force-plot-bars {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .force-bar {
            height: 30px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            border-radius: 5px;
            color: white;
            font-size: 12px;
            transition: all 0.3s ease;
        }
        .force-bar:hover {
            opacity: 0.8;
            transform: translateX(5px);
        }
        .force-label {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
    </style>`;
    
    container.innerHTML = html;
}

/**
 * Render SHAP waterfall chart
 */
function renderSHAPWaterfallChart(data) {
    const baseValue = data.base_value || 0.5;
    const prediction = data.prediction || 0.65;
    const shapValues = data.shap_values;
    
    // Sort features by absolute SHAP value
    const features = Object.keys(shapValues);
    const sorted = features.map(f => ({ feature: f, value: shapValues[f] }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 10);
    
    // Calculate cumulative values
    const labels = ['Base Value', ...sorted.map(s => s.feature), 'Prediction'];
    const values = [baseValue];
    let cumulative = baseValue;
    
    sorted.forEach(item => {
        cumulative += item.value;
        values.push(cumulative);
    });
    values.push(prediction);
    
    // Create waterfall data
    const waterfallData = [{
        type: 'waterfall',
        orientation: 'v',
        x: labels,
        y: [baseValue, ...sorted.map(s => s.value), 0],
        measure: ['absolute', ...sorted.map(() => 'relative'), 'total'],
        text: labels.map((l, i) => i === 0 ? formatNumber(baseValue, 3) : 
                                    i === labels.length - 1 ? formatNumber(prediction, 3) :
                                    formatNumber(sorted[i-1].value, 3)),
        textposition: 'outside',
        connector: {
            line: {
                color: 'rgb(63, 63, 63)'
            }
        },
        increasing: { marker: { color: 'rgba(25, 135, 84, 0.7)' } },
        decreasing: { marker: { color: 'rgba(220, 53, 69, 0.7)' } },
        totals: { marker: { color: 'rgba(13, 110, 253, 0.7)' } }
    }];
    
    const layout = {
        title: {
            text: 'SHAP Waterfall Chart',
            font: { size: 18 }
        },
        xaxis: {
            title: 'Features',
            tickangle: -45
        },
        yaxis: {
            title: 'Cumulative SHAP Value'
        },
        margin: { t: 50, b: 120, l: 80, r: 50 },
        showlegend: false
    };
    
    Plotly.newPlot('shapWaterfallChart', waterfallData, layout, { responsive: true });
}

/**
 * Render influential features table
 */
function renderInfluentialFeaturesTable(data) {
    const tbody = document.getElementById('influentialFeaturesBody');
    tbody.innerHTML = '';
    
    const shapValues = data.shap_values;
    const featureValues = data.feature_values || {};
    
    // Sort by absolute SHAP value
    const features = Object.keys(shapValues);
    const sorted = features.map(f => ({
        feature: f,
        shapValue: shapValues[f],
        featureValue: featureValues[f] || 0
    })).sort((a, b) => Math.abs(b.shapValue) - Math.abs(a.shapValue))
      .slice(0, 10);
    
    sorted.forEach((item, index) => {
        const row = document.createElement('tr');
        
        const impact = item.shapValue > 0 ? 'Positive' : 'Negative';
        const impactClass = item.shapValue > 0 ? 'text-success' : 'text-danger';
        const impactIcon = item.shapValue > 0 ? 'arrow-up' : 'arrow-down';
        
        row.innerHTML = `
            <td><span class="badge bg-primary">${index + 1}</span></td>
            <td><strong>${item.feature}</strong></td>
            <td>${formatNumber(item.shapValue, 4)}</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar ${item.shapValue > 0 ? 'bg-success' : 'bg-danger'}" 
                         style="width: ${Math.abs(item.shapValue) * 100}%">
                        ${formatPercentage(Math.abs(item.shapValue))}
                    </div>
                </div>
            </td>
            <td class="${impactClass}">
                <i class="bi bi-${impactIcon}-circle-fill"></i> ${impact}
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

/**
 * Export functions
 */
function downloadFeatureImportance() {
    if (currentExplanationData) {
        downloadJSON(currentExplanationData, 'feature_importance.json');
    }
}

function exportExplanationPDF() {
    alert('PDF export functionality would require a backend service or library like jsPDF');
}

function exportExplanationPNG() {
    // Export all charts as PNG
    if (featureImportanceChart) {
        downloadChartAsPNG('featureImportanceChart', 'feature_importance.png');
    }
}

function exportExplanationJSON() {
    if (currentExplanationData) {
        downloadJSON(currentExplanationData, 'shap_explanation.json');
    }
}
