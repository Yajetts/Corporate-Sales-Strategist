/**
 * Business Optimization View JavaScript
 */

let currentOptimizationData = null;
let resourceDistributionChart = null;
let scenarioComparisonChart = null;

// Update what-if sliders
const whatIfSliders = ['whatIfBudget', 'whatIfCapacity', 'whatIfDemand'];
whatIfSliders.forEach(sliderId => {
    const slider = document.getElementById(sliderId);
    const valueDisplay = document.getElementById(sliderId + 'Value');
    
    slider.addEventListener('input', function(e) {
        valueDisplay.textContent = e.target.value;
    });
});

// Form submission
document.getElementById('optimizationForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    try {
        showLoading('alertContainer');
        
        // Get products data
        const productsDataStr = document.getElementById('productsData').value;
        
        if (!productsDataStr.trim()) {
            showError('alertContainer', 'Please enter products data');
            return;
        }
        
        let products;
        try {
            products = JSON.parse(productsDataStr);
        } catch (error) {
            showError('alertContainer', 'Invalid JSON format', error.message);
            return;
        }
        
        // Get constraints
        const constraints = {
            budget_limit: parseFloat(document.getElementById('budgetLimit').value),
            capacity_limit: parseFloat(document.getElementById('capacityLimit').value)
        };
        
        // Get objectives
        const objectives = {
            maximize_revenue: document.getElementById('maxRevenue').checked,
            minimize_cost: document.getElementById('minCost').checked
        };
        
        // Call API
        const result = await apiClient.businessOptimization(products, { constraints, objectives });
        
        // Display results
        displayResults(result);
        
        showSuccess('alertContainer', 'Optimization completed successfully!');
        
    } catch (error) {
        showError('alertContainer', 'Failed to optimize portfolio', error.message);
    }
});

/**
 * Load sample portfolio
 */
async function loadSamplePortfolio() {
    const sampleProducts = [
        { name: 'Product A', demand: 1000, cost: 50, price: 80, min_production: 100 },
        { name: 'Product B', demand: 800, cost: 40, price: 70, min_production: 50 },
        { name: 'Product C', demand: 1200, cost: 60, price: 95, min_production: 150 },
        { name: 'Product D', demand: 600, cost: 35, price: 55, min_production: 50 },
        { name: 'Product E', demand: 900, cost: 45, price: 75, min_production: 100 }
    ];
    
    document.getElementById('productsData').value = JSON.stringify(sampleProducts, null, 2);
    
    showSuccess('alertContainer', 'Sample portfolio loaded. Click "Optimize Portfolio" to analyze.');
}

/**
 * Display optimization results
 */
function displayResults(data) {
    currentOptimizationData = data;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Use production_priorities or recommendations (backward compatibility)
    const recommendations = data.production_priorities || data.recommendations || [];
    
    // Calculate metrics
    const totalRevenue = recommendations.reduce((sum, p) => sum + (p.quantity * p.price), 0);
    const totalCost = recommendations.reduce((sum, p) => sum + (p.quantity * p.cost), 0);
    const netProfit = totalRevenue - totalCost;
    const roi = totalCost > 0 ? (netProfit / totalCost) : 0;
    
    // Update summary metrics
    document.getElementById('totalRevenue').textContent = '$' + formatNumber(totalRevenue, 0);
    document.getElementById('totalCost').textContent = '$' + formatNumber(totalCost, 0);
    document.getElementById('netProfit').textContent = '$' + formatNumber(netProfit, 0);
    document.getElementById('roi').textContent = formatPercentage(roi);
    
    // Render visualizations
    renderPriorityMatrix(recommendations);
    renderProductionAllocationChart(recommendations);
    renderResourceDistributionChart(recommendations);
    renderProductionTable(recommendations);
    renderScenarioComparison(data);
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Render priority matrix with Plotly
 */
function renderPriorityMatrix(recommendations) {
    // Create 2D matrix based on profit margin and demand
    const products = recommendations.map(r => r.name);
    const profitMargins = recommendations.map(r => ((r.price - r.cost) / r.price));
    const demands = recommendations.map(r => r.demand);
    
    const matrixData = [{
        x: products,
        y: ['Profit Margin', 'Demand', 'Priority Score'],
        z: [
            profitMargins,
            demands.map(d => d / Math.max(...demands)),
            recommendations.map(r => r.priority_score || 0.5)
        ],
        type: 'heatmap',
        colorscale: [
            [0, 'rgb(220, 53, 69)'],
            [0.5, 'rgb(255, 193, 7)'],
            [1, 'rgb(25, 135, 84)']
        ],
        hoverongaps: false,
        hovertemplate: '%{y}: %{z:.2f}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'Product Priority Analysis',
            font: { size: 18 }
        },
        xaxis: {
            title: 'Products'
        },
        yaxis: {
            title: 'Metrics'
        },
        margin: { t: 50, b: 80, l: 100, r: 50 }
    };
    
    Plotly.newPlot('priorityMatrix', matrixData, layout, { responsive: true });
}

/**
 * Render production allocation chart
 */
function renderProductionAllocationChart(recommendations) {
    const pieData = [{
        values: recommendations.map(r => r.quantity),
        labels: recommendations.map(r => r.name),
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: generateColors(recommendations.length)
        },
        textinfo: 'label+percent',
        textposition: 'outside',
        hovertemplate: '%{label}: %{value} units<br>%{percent}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'Production Quantity Distribution',
            font: { size: 16 }
        },
        showlegend: true,
        legend: {
            orientation: 'v',
            x: 1.1,
            y: 0.5
        },
        margin: { t: 50, b: 50, l: 50, r: 150 }
    };
    
    Plotly.newPlot('productionAllocationChart', pieData, layout, { responsive: true });
}

/**
 * Render resource distribution chart
 */
function renderResourceDistributionChart(recommendations) {
    const ctx = document.getElementById('resourceDistributionChart').getContext('2d');
    
    if (resourceDistributionChart) {
        resourceDistributionChart.destroy();
    }
    
    const products = recommendations.map(r => r.name);
    const costs = recommendations.map(r => r.quantity * r.cost);
    const revenues = recommendations.map(r => r.quantity * r.price);
    
    resourceDistributionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: products,
            datasets: [
                {
                    label: 'Production Cost',
                    data: costs,
                    backgroundColor: 'rgba(220, 53, 69, 0.7)',
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Expected Revenue',
                    data: revenues,
                    backgroundColor: 'rgba(25, 135, 84, 0.7)',
                    borderColor: 'rgba(25, 135, 84, 1)',
                    borderWidth: 2
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
                    text: 'Cost vs Revenue by Product'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += '$' + formatNumber(context.parsed.y, 0);
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
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
 * Render production table
 */
function renderProductionTable(recommendations) {
    const tbody = document.getElementById('productionTableBody');
    tbody.innerHTML = '';
    
    // Sort by priority score
    const sorted = [...recommendations].sort((a, b) => 
        (b.priority_score || 0) - (a.priority_score || 0)
    );
    
    sorted.forEach(product => {
        const row = document.createElement('tr');
        
        const revenue = product.quantity * product.price;
        const cost = product.quantity * product.cost;
        const profitMargin = ((product.price - product.cost) / product.price);
        const priority = getPriorityBadge(product.priority_score || 0.5);
        
        row.innerHTML = `
            <td><strong>${product.name}</strong></td>
            <td>${formatNumber(product.quantity, 0)} units</td>
            <td>$${formatNumber(revenue, 0)}</td>
            <td>$${formatNumber(cost, 0)}</td>
            <td>${formatPercentage(profitMargin)}</td>
            <td>${priority}</td>
        `;
        
        tbody.appendChild(row);
    });
}

/**
 * Get priority badge
 */
function getPriorityBadge(score) {
    if (score >= 0.8) return '<span class="badge bg-danger">High</span>';
    if (score >= 0.5) return '<span class="badge bg-warning">Medium</span>';
    return '<span class="badge bg-secondary">Low</span>';
}

/**
 * Render scenario comparison
 */
function renderScenarioComparison(data) {
    const ctx = document.getElementById('scenarioComparisonChart').getContext('2d');
    
    if (scenarioComparisonChart) {
        scenarioComparisonChart.destroy();
    }
    
    // Generate comparison scenarios
    const scenarios = ['Current', 'Conservative', 'Aggressive', 'Balanced'];
    const revenues = [100, 85, 120, 105].map(v => v * 10000);
    const costs = [60, 50, 80, 65].map(v => v * 10000);
    const profits = revenues.map((r, i) => r - costs[i]);
    
    scenarioComparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: scenarios,
            datasets: [
                {
                    label: 'Revenue',
                    data: revenues,
                    backgroundColor: 'rgba(25, 135, 84, 0.7)',
                    borderColor: 'rgba(25, 135, 84, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Cost',
                    data: costs,
                    backgroundColor: 'rgba(220, 53, 69, 0.7)',
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Profit',
                    data: profits,
                    backgroundColor: 'rgba(13, 110, 253, 0.7)',
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 2
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
                    text: 'Scenario Comparison Analysis'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += '$' + formatNumber(context.parsed.y, 0);
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
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
 * Run what-if analysis
 */
async function runWhatIfAnalysis() {
    if (!currentOptimizationData) {
        showError('alertContainer', 'Please run optimization first');
        return;
    }
    
    try {
        showLoading('alertContainer');
        
        // Get adjustments
        const budgetAdj = parseFloat(document.getElementById('whatIfBudget').value) / 100;
        const capacityAdj = parseFloat(document.getElementById('whatIfCapacity').value) / 100;
        const demandAdj = parseFloat(document.getElementById('whatIfDemand').value) / 100;
        
        // Adjust constraints
        const originalBudget = parseFloat(document.getElementById('budgetLimit').value);
        const originalCapacity = parseFloat(document.getElementById('capacityLimit').value);
        
        const newConstraints = {
            budget_limit: originalBudget * (1 + budgetAdj),
            capacity_limit: originalCapacity * (1 + capacityAdj)
        };
        
        // Adjust products data
        const productsDataStr = document.getElementById('productsData').value;
        const products = JSON.parse(productsDataStr);
        const adjustedProducts = products.map(p => ({
            ...p,
            demand: p.demand * (1 + demandAdj)
        }));
        
        // Get objectives
        const objectives = {
            maximize_revenue: document.getElementById('maxRevenue').checked,
            minimize_cost: document.getElementById('minCost').checked
        };
        
        // Call API with adjusted parameters
        const result = await apiClient.businessOptimization(adjustedProducts, { 
            constraints: newConstraints, 
            objectives 
        });
        
        // Display results
        displayResults(result);
        
        showSuccess('alertContainer', 'What-if analysis completed successfully!');
        
    } catch (error) {
        showError('alertContainer', 'Failed to run what-if analysis', error.message);
    }
}

/**
 * Download priority matrix
 */
function downloadPriorityMatrix() {
    if (currentOptimizationData) {
        downloadJSON(currentOptimizationData, 'optimization_results.json');
    }
}

/**
 * Generate colors for charts
 */
function generateColors(count) {
    const colors = [];
    for (let i = 0; i < count; i++) {
        const hue = (i * 360 / count) % 360;
        colors.push(`hsl(${hue}, 70%, 60%)`);
    }
    return colors;
}
