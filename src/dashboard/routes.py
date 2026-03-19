"""Dashboard routes for the Sales Strategist System"""

import logging
from datetime import datetime, timezone
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, current_app

from src.post_analysis.storage.artifact_store import ArtifactStore
from src.post_analysis.rendering import markdown_to_html

logger = logging.getLogger(__name__)

# Avoid unlocking the UI due to pre-existing artifacts on disk.
_DASHBOARD_STARTED_AT = datetime.now(timezone.utc)

# Create blueprint
dashboard_bp = Blueprint(
    'dashboard',
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/dashboard/static'
)


@dashboard_bp.app_context_processor
def _inject_analysis_overview_ready():
    """Make `analysis_overview_ready` available across all dashboard templates."""

    try:
        ready = _analysis_overview_ready()
    except Exception as e:
        logger.warning(f"Failed computing analysis_overview_ready for template context: {e}")
        ready = False

    return {"analysis_overview_ready": ready}


@dashboard_bp.route('/')
def index():
    """
    Dashboard home page.
    """
    overview_locked = (request.args.get("overview") or "").strip().lower() == "locked"
    return render_template(
        'index.html',
        overview_locked=overview_locked,
    )


@dashboard_bp.route('/market-analysis')
def market_analysis():
    """
    Market Analysis view.
    """
    return render_template('market_analysis.html')


@dashboard_bp.route('/enterprise')
def enterprise():
    """Enterprise Analyst view."""
    return render_template('enterprise.html')


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


@dashboard_bp.route('/final-insights')
def analysis_overview_redirect():
    """Backward-compatible redirect to Analysis Overview."""
    return redirect(url_for('dashboard.analysis_overview'))


@dashboard_bp.route('/analysis-overview')
def analysis_overview():
    """Analysis Overview view (latest run only)."""

    if not _analysis_overview_ready():
        return redirect(url_for('dashboard.index', overview='locked'))

    store = ArtifactStore()

    # Latest run only: ignore any run_id query params.
    run_id = store.get_latest_run()

    selected = None
    report_html = None

    if run_id:
        paths = store.paths(run_id)
        status = store.get_status(run_id)
        selected = {
            'run_id': run_id,
            'status': status,
            'has_report': paths.report_md_path.exists(),
            'has_audio': paths.audio_mp3_path.exists(),
            'has_podcast_script': paths.podcast_script_path.exists(),
            'has_pdf': paths.report_pdf_path.exists(),
        }

        if paths.report_md_path.exists():
            md = paths.report_md_path.read_text(encoding='utf-8')
            try:
                report_html = markdown_to_html(md)
            except Exception as e:
                logger.warning(f"Markdown rendering unavailable: {e}")
                # Preserve line breaks but keep the dashboard's normal font.
                safe = md.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                report_html = '<div style="white-space: pre-wrap; font-family: inherit;">' + safe + '</div>'

    debug = (request.args.get("debug") or "").strip().lower() in {"1", "true", "yes"}

    return render_template(
        'analysis_overview.html',
        selected=selected,
        report_html=report_html,
        debug=debug,
        api_prefix='/api/v1'
    )


def _analysis_overview_ready() -> bool:
    """True only when a *complete* module workflow has executed in this session.

    Definition of "complete": every module has produced a non-empty output AND
    that output's timestamp is >= the time the dashboard process started.

    This avoids unlocking from stale DB/cache data while still allowing the
    button to appear immediately after the user runs all modules.
    """

    def _non_empty_payload(payload: object) -> bool:
        if payload is None:
            return False
        if isinstance(payload, str):
            return bool(payload.strip())
        if isinstance(payload, (list, tuple, set, dict)):
            return len(payload) > 0
        return True

    try:
        # Use the local output cache written by the API routes when each module runs.
        # This keeps the UI rule aligned with the user's workflow and avoids using
        # stale DB rows from previous runs.
        from src.post_analysis.storage.module_output_store import ModuleOutputStore

        expected_boot_id = None
        try:
            expected_boot_id = current_app.config.get("BOOT_ID")
        except Exception:
            expected_boot_id = None

        store = ModuleOutputStore()
        required_modules = ["enterprise", "market", "strategy", "performance", "optimization", "shap"]
        for module in required_modules:
            rec = store.read_latest(module)
            if not rec:
                return False
            if not _non_empty_payload(rec.payload):
                return False
            if expected_boot_id and str(rec.payload.get("_boot_id") or "") != str(expected_boot_id):
                return False
        return True
    except Exception as e:
        logger.warning(f"Failed reading cached module outputs for readiness check: {e}")
        return False
