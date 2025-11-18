"""Dashboard routes for the Sales Strategist System"""

import logging
from flask import Blueprint, render_template, request, jsonify

logger = logging.getLogger(__name__)

# Create blueprint
dashboard_bp = Blueprint(
    'dashboard',
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/dashboard/static'
)


@dashboard_bp.route('/')
def index():
    """
    Dashboard home page.
    """
    return render_template('index.html')


@dashboard_bp.route('/market-analysis')
def market_analysis():
    """
    Market Analysis view.
    """
    return render_template('market_analysis.html')


@dashboard_bp.route('/strategy')
def strategy():
    """
    Strategy Generation view.
    """
    return render_template('strategy.html')


@dashboard_bp.route('/performance')
def performance():
    """
    Performance Monitoring view.
    """
    return render_template('performance.html')


@dashboard_bp.route('/explainability')
def explainability():
    """
    Model Explainability view.
    """
    return render_template('explainability.html')


@dashboard_bp.route('/business-optimization')
def business_optimization():
    """
    Business Optimization view.
    """
    return render_template('business_optimization.html')
