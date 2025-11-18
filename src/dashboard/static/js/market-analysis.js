/**
 * Market Analysis View JavaScript
 */

let currentAnalysisData = null;
let clusterChart = null;
let clusterSizeChart = null;
let clusterMetricsChart = null;

/**
 * Check if a library is loaded
 * @param {string} libraryName - Name of the global library object
 * @returns {boolean} - True if library is loaded
 */
function isLibraryLoaded(libraryName) {
    return typeof window[libraryName] !== 'undefined';
}

/**
 * Wait for library to load with timeout
 * @param {string} libraryName - Name of the global library object
 * @param {number} timeout - Maximum wait time in milliseconds
 * @returns {Promise<boolean>} - Resolves when library is loaded or timeout
 */
function waitForLibrary(libraryName, timeout = 5000) {
    return new Promise((resolve) => {
        if (isLibraryLoaded(libraryName)) {
            resolve(true);
            return;
        }
        
        const startTime = Date.now();
        const interval = setInterval(() => {
            if (isLibraryLoaded(libraryName)) {
                clearInterval(interval);
                resolve(true);
            } else if (Date.now() - startTime >= timeout) {
                clearInterval(interval);
                resolve(false);
            }
        }, 100);
    });
}

/**
 * Check all required libraries
 * @param {string[]} libraries - Array of library names
 * @returns {Object} - Status of each library
 */
async function checkRequiredLibraries(libraries) {
    const status = {};
    
    for (const lib of libraries) {
        const loaded = await waitForLibrary(lib);
        status[lib] = loaded;
    }
    
    return status;
}

/**
 * Initialize the market analysis module
 * Checks library availability before setting up event handlers
 */
async function initializeMarketAnalysis() {
    console.log('Initializing Market Analysis module...');
    
    // Check required libraries
    const requiredLibraries = ['d3', 'Chart', 'Plotly'];
    const libraryStatus = await checkRequiredLibraries(requiredLibraries);
    
    console.log('Library status:', libraryStatus);
    
    // Check if any libraries failed to load
    const failedLibraries = Object.keys(libraryStatus).filter(lib => !libraryStatus[lib]);
    
    if (failedLibraries.length > 0) {
        const errorMessage = `Failed to load required visualization libraries: ${failedLibraries.join(', ')}. Please refresh the page.`;
        console.error(errorMessage);
        showError('alertContainer', 'Visualization libraries failed to load', errorMessage);
        return;
    }
    
    console.log('All libraries loaded successfully');
    
    // Set up event handlers
    setupEventHandlers();
}

/**
 * Set up event handlers for the page
 */
function setupEventHandlers() {
    // Update similarity threshold display
    const similarityThreshold = document.getElementById('similarityThreshold');
    if (similarityThreshold) {
        similarityThreshold.addEventListener('input', function(e) {
            document.getElementById('similarityValue').textContent = e.target.value;
        });
    }
    
    // Form submission
    const marketAnalysisForm = document.getElementById('marketAnalysisForm');
    if (marketAnalysisForm) {
        marketAnalysisForm.addEventListener('submit', handleFormSubmit);
    }
}

/**
 * Handle form submission
 */
async function handleFormSubmit(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('marketDataFile');
    
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('alertContainer', 'No file selected', 'Please select a CSV or JSON file to upload');
        return;
    }
    
    const file = fileInput.files[0];
    
    try {
        showLoading('alertContainer');
        
        console.log(`Reading file: ${file.name} (${file.size} bytes)`);
        
        // Read file
        const marketData = await readFile(file);
        
        if (!marketData || (Array.isArray(marketData) && marketData.length === 0)) {
            throw new Error('File contains no data or is empty');
        }
        
        console.log(`File read successfully, ${Array.isArray(marketData) ? marketData.length : 'unknown'} entities`);
        
        // Get options
        const options = {
            autoSelectClusters: document.getElementById('autoSelectClusters').value === 'true',
            similarityThreshold: parseFloat(document.getElementById('similarityThreshold').value),
            topKLinks: parseInt(document.getElementById('topKLinks').value)
        };
        
        console.log('Calling market analysis API with options:', options);
        
        // Call API
        const result = await apiClient.marketAnalysis(marketData, options);
        
        if (!result) {
            throw new Error('API returned empty result');
        }
        
        console.log('API call successful, displaying results...');
        
        // Display results
        displayResults(result);
        
        showSuccess('alertContainer', 'Market analysis completed successfully!');
        
    } catch (error) {
        console.error('Form submission error:', error);
        
        // Provide specific error messages based on error type
        let userMessage = 'Failed to analyze market data';
        let technicalDetails = error.message;
        
        if (error.message && error.message.includes('Unsupported file format')) {
            userMessage = 'Unsupported file format';
            technicalDetails = 'Please upload a CSV or JSON file';
        } else if (error.message && error.message.includes('Failed to read file')) {
            userMessage = 'Failed to read file';
            technicalDetails = 'The file could not be read. Please check the file format and try again.';
        } else if (error.message && error.message.includes('empty')) {
            userMessage = 'Empty file';
            technicalDetails = 'The uploaded file contains no data. Please check the file and try again.';
        } else if (error.message && error.message.includes('validation')) {
            userMessage = 'Invalid data received';
            technicalDetails = 'The API returned invalid data. ' + error.message;
        } else if (error.status) {
            userMessage = `API Error (${error.status})`;
            technicalDetails = error.message || 'The server returned an error. Please try again.';
        }
        
        showError('alertContainer', userMessage, technicalDetails);
    } finally {
        // Ensure loading state is cleared
        hideLoading('alertContainer');
    }
}

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
                    resolve(parseCSV(content));
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
 * Parse CSV content
 */
function parseCSV(content) {
    const lines = content.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        
        for (let j = 0; j < headers.length; j++) {
            const value = values[j].trim();
            // Try to parse as number
            row[headers[j]] = isNaN(value) ? value : parseFloat(value);
        }
        
        data.push(row);
    }
    
    return data;
}

/**
 * Load sample data
 */
async function loadSampleData() {
    try {
        showLoading('alertContainer');
        
        // Generate sample market data
        console.log('Generating sample market data...');
        const sampleData = generateSampleMarketData(100);
        
        if (!sampleData || !Array.isArray(sampleData) || sampleData.length === 0) {
            throw new Error('Failed to generate sample data');
        }
        
        console.log(`Generated ${sampleData.length} sample entities`);
        
        const options = {
            autoSelectClusters: true,
            similarityThreshold: 0.7,
            topKLinks: 20
        };
        
        console.log('Calling market analysis API...');
        const result = await apiClient.marketAnalysis(sampleData, options);
        
        if (!result) {
            throw new Error('API returned empty result');
        }
        
        console.log('API call successful, displaying results...');
        displayResults(result);
        
        showSuccess('alertContainer', 'Sample data loaded and analyzed successfully!');
        
    } catch (error) {
        console.error('Sample data loading error:', error);
        
        // Provide more specific error messages
        let userMessage = 'Failed to load sample data';
        let technicalDetails = error.message;
        
        if (error.message && error.message.includes('API')) {
            userMessage = 'Failed to analyze sample data';
            technicalDetails = 'The API request failed. ' + error.message;
        } else if (error.message && error.message.includes('generate')) {
            userMessage = 'Failed to generate sample data';
            technicalDetails = 'Sample data generation failed. ' + error.message;
        } else if (error.message && error.message.includes('validation')) {
            userMessage = 'Invalid data received from API';
            technicalDetails = 'Data validation failed. ' + error.message;
        }
        
        showError('alertContainer', userMessage, technicalDetails);
    } finally {
        // Ensure loading state is cleared
        hideLoading('alertContainer');
    }
}

// Make functions globally accessible for onclick handlers
window.loadSampleData = loadSampleData;
window.resetGraphZoom = resetGraphZoom;
window.downloadGraphData = downloadGraphData;
window.downloadChartAsPNG = downloadChartAsPNG;

/**
 * Generate sample market data
 */
function generateSampleMarketData(count) {
    if (typeof count !== 'number' || count <= 0 || count > 10000) {
        throw new Error('Invalid count: must be a number between 1 and 10000');
    }
    
    const data = [];
    
    try {
        for (let i = 0; i < count; i++) {
            data.push({
                revenue: Math.random() * 1000000,
                employees: Math.floor(Math.random() * 1000),
                market_share: Math.random(),
                growth_rate: (Math.random() - 0.5) * 0.5,
                customer_satisfaction: Math.random(),
                innovation_score: Math.random()
            });
        }
    } catch (error) {
        console.error('Error generating sample data:', error);
        throw new Error('Failed to generate sample market data: ' + error.message);
    }
    
    return data;
}

/**
 * Validate API response structure
 * @param {Object} data - API response data
 * @returns {Object} - Validation result with isValid flag and errors array
 */
function validateAnalysisData(data) {
    const errors = [];
    
    // Check if data exists
    if (!data || typeof data !== 'object') {
        errors.push('Invalid data: response is null or not an object');
        return { isValid: false, errors };
    }
    
    // Validate num_entities
    if (typeof data.num_entities !== 'number' || data.num_entities < 0) {
        errors.push('Invalid num_entities: must be a non-negative number');
    }
    
    // Validate clusters object
    if (!data.clusters || typeof data.clusters !== 'object') {
        errors.push('Invalid clusters: must be an object');
    } else {
        // Validate n_clusters
        if (typeof data.clusters.n_clusters !== 'number' || data.clusters.n_clusters < 0) {
            errors.push('Invalid clusters.n_clusters: must be a non-negative number');
        }
        
        // Validate labels array
        if (!Array.isArray(data.clusters.labels)) {
            errors.push('Invalid clusters.labels: must be an array');
        } else if (data.clusters.labels.length === 0) {
            errors.push('Invalid clusters.labels: array is empty');
        } else if (data.clusters.labels.length !== data.num_entities) {
            errors.push(`Invalid clusters.labels: length (${data.clusters.labels.length}) does not match num_entities (${data.num_entities})`);
        }
        
        // Validate metrics (optional but if present, must be valid)
        if (data.clusters.metrics && typeof data.clusters.metrics !== 'object') {
            errors.push('Invalid clusters.metrics: must be an object');
        }
    }
    
    // Validate graph object
    if (!data.graph || typeof data.graph !== 'object') {
        errors.push('Invalid graph: must be an object');
    } else {
        // Validate num_nodes
        if (typeof data.graph.num_nodes !== 'number' || data.graph.num_nodes < 0) {
            errors.push('Invalid graph.num_nodes: must be a non-negative number');
        }
        
        // Validate num_edges
        if (typeof data.graph.num_edges !== 'number' || data.graph.num_edges < 0) {
            errors.push('Invalid graph.num_edges: must be a non-negative number');
        }
        
        // Validate predicted_links array
        if (!Array.isArray(data.graph.predicted_links)) {
            errors.push('Invalid graph.predicted_links: must be an array');
        } else {
            // Validate each link
            data.graph.predicted_links.forEach((link, index) => {
                if (typeof link.source !== 'number' || typeof link.target !== 'number') {
                    errors.push(`Invalid link at index ${index}: source and target must be numbers`);
                }
                if (typeof link.score !== 'number' || link.score < 0) {
                    errors.push(`Invalid link at index ${index}: score must be a non-negative number`);
                }
            });
        }
    }
    
    // Validate potential_clients array
    if (!Array.isArray(data.potential_clients)) {
        errors.push('Invalid potential_clients: must be an array');
    } else if (data.potential_clients.length > 0) {
        // Validate each client
        data.potential_clients.forEach((client, index) => {
            if (typeof client.entity_id === 'undefined') {
                errors.push(`Invalid client at index ${index}: missing entity_id`);
            }
            if (typeof client.cluster_id === 'undefined') {
                errors.push(`Invalid client at index ${index}: missing cluster_id`);
            }
            if (typeof client.connectivity_score !== 'number' || client.connectivity_score < 0) {
                errors.push(`Invalid client at index ${index}: connectivity_score must be a non-negative number`);
            }
        });
    }
    
    return {
        isValid: errors.length === 0,
        errors
    };
}

/**
 * Safely render a visualization with error handling
 * @param {string} visualizationName - Name of the visualization for error messages
 * @param {Function} renderFunction - Function to execute for rendering
 * @returns {boolean} - Success status
 */
function safeRenderVisualization(visualizationName, renderFunction) {
    try {
        console.log(`Rendering ${visualizationName}...`);
        renderFunction();
        console.log(`✓ ${visualizationName} rendered successfully`);
        return true;
    } catch (error) {
        console.error(`✗ Failed to render ${visualizationName}:`, error);
        
        // Log detailed error for debugging
        console.error('Error details:', {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
        
        // Show user-friendly error message
        const errorContainer = document.getElementById('alertContainer');
        if (errorContainer) {
            const existingAlert = errorContainer.querySelector('.alert');
            if (!existingAlert) {
                showError(
                    'alertContainer',
                    `Failed to render ${visualizationName}`,
                    'See console for technical details. Other visualizations may still work.'
                );
            }
        }
        
        return false;
    }
}

/**
 * Display analysis results
 */
function displayResults(data) {
    // Validate data structure
    const validation = validateAnalysisData(data);
    
    if (!validation.isValid) {
        console.error('Data validation failed:', validation.errors);
        showError(
            'alertContainer',
            'Invalid data received from server',
            'Data validation errors: ' + validation.errors.join('; ')
        );
        return;
    }
    
    console.log('Data validation passed');
    currentAnalysisData = data;
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'block';
    
    // Update summary cards
    document.getElementById('totalEntities').textContent = formatNumber(data.num_entities, 0);
    document.getElementById('totalClusters').textContent = formatNumber(data.clusters.n_clusters, 0);
    document.getElementById('graphNodes').textContent = formatNumber(data.graph.num_nodes, 0);
    document.getElementById('graphEdges').textContent = formatNumber(data.graph.num_edges, 0);
    
    // Render visualizations with error handling
    safeRenderVisualization('Cluster Visualization', () => renderClusterVisualization(data.clusters));
    safeRenderVisualization('Cluster Size Chart', () => renderClusterSizeChart(data.clusters));
    safeRenderVisualization('Cluster Metrics Chart', () => renderClusterMetricsChart(data.clusters));
    safeRenderVisualization('Relationship Graph', () => renderGraphVisualization(data.graph));
    safeRenderVisualization('Potential Clients Table', () => renderPotentialClientsTable(data.potential_clients));
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

/**
 * Render cluster visualization with Plotly
 */
function renderClusterVisualization(clusters) {
    // Validate Plotly is available
    if (typeof Plotly === 'undefined') {
        throw new Error('Plotly library is not loaded');
    }
    
    if (!clusters || !clusters.labels || !Array.isArray(clusters.labels)) {
        throw new Error('Invalid cluster data: labels array is missing or invalid');
    }
    
    const labels = clusters.labels;
    const nClusters = clusters.n_clusters;
    
    // Count entities per cluster
    const clusterCounts = {};
    labels.forEach(label => {
        clusterCounts[label] = (clusterCounts[label] || 0) + 1;
    });
    
    // Prepare data for pie chart
    const pieData = [{
        values: Object.values(clusterCounts),
        labels: Object.keys(clusterCounts).map(c => `Cluster ${c}`),
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: generateColors(nClusters)
        },
        textinfo: 'label+percent',
        textposition: 'outside'
    }];
    
    const layout = {
        title: {
            text: 'Customer Segment Distribution',
            font: { size: 18 }
        },
        showlegend: true,
        legend: {
            orientation: 'v',
            x: 1.1,
            y: 0.5
        },
        margin: { t: 50, b: 50, l: 50, r: 150 }
    };
    
    Plotly.newPlot('clusterChart', pieData, layout, { responsive: true });
}

/**
 * Render cluster size chart with Chart.js
 */
function renderClusterSizeChart(clusters) {
    // Validate Chart.js is available
    if (typeof Chart === 'undefined') {
        throw new Error('Chart.js library is not loaded');
    }
    
    if (!clusters || !clusters.labels || !Array.isArray(clusters.labels)) {
        throw new Error('Invalid cluster data: labels array is missing or invalid');
    }
    
    const labels = clusters.labels;
    
    // Count entities per cluster
    const clusterCounts = {};
    labels.forEach(label => {
        clusterCounts[label] = (clusterCounts[label] || 0) + 1;
    });
    
    const ctx = document.getElementById('clusterSizeChart').getContext('2d');
    
    if (clusterSizeChart) {
        clusterSizeChart.destroy();
    }
    
    clusterSizeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(clusterCounts).map(c => `Cluster ${c}`),
            datasets: [{
                label: 'Number of Entities',
                data: Object.values(clusterCounts),
                backgroundColor: 'rgba(25, 135, 84, 0.7)',
                borderColor: 'rgba(25, 135, 84, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Entities per Cluster'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            }
        }
    });
}

/**
 * Render cluster metrics chart
 */
function renderClusterMetricsChart(clusters) {
    // Validate Chart.js is available
    if (typeof Chart === 'undefined') {
        throw new Error('Chart.js library is not loaded');
    }
    
    const canvas = document.getElementById('clusterMetricsChart');
    if (!canvas) {
        throw new Error('Cluster metrics chart canvas not found');
    }
    
    const ctx = canvas.getContext('2d');
    
    if (clusterMetricsChart) {
        clusterMetricsChart.destroy();
    }
    
    // Extract metrics if available
    const metrics = clusters.metrics || {};
    const metricNames = Object.keys(metrics);
    const metricValues = Object.values(metrics);
    
    if (metricNames.length === 0) {
        // Show placeholder
        document.getElementById('clusterMetricsChart').parentElement.innerHTML = 
            '<p class="text-muted text-center">No metrics available</p>';
        return;
    }
    
    clusterMetricsChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: metricNames,
            datasets: [{
                label: 'Cluster Quality Metrics',
                data: metricValues,
                backgroundColor: 'rgba(13, 202, 240, 0.2)',
                borderColor: 'rgba(13, 202, 240, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(13, 202, 240, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(13, 202, 240, 1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

/**
 * Render graph visualization with D3.js
 */
function renderGraphVisualization(graph) {
    // Validate D3.js is available
    if (typeof d3 === 'undefined') {
        throw new Error('D3.js library is not loaded');
    }
    
    const container = document.getElementById('graphVisualization');
    if (!container) {
        throw new Error('Graph visualization container not found');
    }
    
    container.innerHTML = ''; // Clear previous
    
    const width = container.clientWidth;
    const height = 500;
    
    if (width === 0 || height === 0) {
        throw new Error('Graph container has invalid dimensions');
    }
    
    // Create SVG
    const svg = d3.select('#graphVisualization')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Create zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.1, 10])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    
    svg.call(zoom);
    
    const g = svg.append('g');
    
    // Prepare nodes and links
    const nodes = Array.from({ length: graph.num_nodes }, (_, i) => ({
        id: i,
        group: i % 5 // Assign to groups for coloring
    }));
    
    const links = graph.predicted_links.map(link => ({
        source: link.source,
        target: link.target,
        value: link.score
    }));
    
    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(20));
    
    // Create links
    const link = g.append('g')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', d => d.value)
        .attr('stroke-width', d => Math.sqrt(d.value) * 2);
    
    // Create nodes
    const node = g.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('r', 8)
        .attr('fill', d => d3.schemeCategory10[d.group % 10])
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    // Add tooltips
    node.append('title')
        .text(d => `Node ${d.id}`);
    
    // Update positions on tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
    });
    
    // Drag functions
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    // Store zoom for reset
    window.graphZoom = zoom;
    window.graphSvg = svg;
}

/**
 * Reset graph zoom
 */
function resetGraphZoom() {
    if (window.graphSvg && window.graphZoom) {
        window.graphSvg.transition()
            .duration(750)
            .call(window.graphZoom.transform, d3.zoomIdentity);
    }
}

/**
 * Render potential clients table
 */
function renderPotentialClientsTable(clients) {
    const tbody = document.getElementById('potentialClientsBody');
    if (!tbody) {
        throw new Error('Potential clients table body not found');
    }
    
    tbody.innerHTML = '';
    
    if (!clients || !Array.isArray(clients) || clients.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No potential clients identified</td></tr>';
        return;
    }
    
    // Sort by connectivity score (create copy to avoid mutating original)
    const sortedClients = [...clients].sort((a, b) => b.connectivity_score - a.connectivity_score);
    
    sortedClients.forEach(client => {
        const row = document.createElement('tr');
        
        const priority = getPriority(client.connectivity_score);
        const priorityBadge = `<span class="badge ${priority.class}">${priority.text}</span>`;
        
        row.innerHTML = `
            <td>${client.entity_id}</td>
            <td><span class="badge bg-secondary">Cluster ${client.cluster_id}</span></td>
            <td>${formatNumber(client.connectivity_score, 3)}</td>
            <td>${priorityBadge}</td>
        `;
        
        tbody.appendChild(row);
    });
}

/**
 * Get priority based on connectivity score
 */
function getPriority(score) {
    if (score >= 0.8) return { text: 'High', class: 'bg-danger' };
    if (score >= 0.5) return { text: 'Medium', class: 'bg-warning' };
    return { text: 'Low', class: 'bg-info' };
}

/**
 * Generate colors for clusters
 */
function generateColors(count) {
    const colors = [];
    for (let i = 0; i < count; i++) {
        const hue = (i * 360 / count) % 360;
        colors.push(`hsl(${hue}, 70%, 60%)`);
    }
    return colors;
}

/**
 * Download graph data
 */
function downloadGraphData() {
    if (currentAnalysisData) {
        downloadJSON(currentAnalysisData, 'market_analysis_results.json');
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeMarketAnalysis);
} else {
    // DOM is already loaded
    initializeMarketAnalysis();
}
