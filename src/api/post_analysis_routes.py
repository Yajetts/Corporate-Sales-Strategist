"""API routes for post-analysis (Analysis Overview) artifacts."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from flask import Blueprint, jsonify, request, send_file
from kombu.exceptions import OperationalError

from src.post_analysis.run_id import new_run_id
from src.post_analysis.storage.artifact_store import ArtifactStore
from src.post_analysis.celery_tasks import generate_analysis_overview_async, generate_podcast_summary_async
from src.post_analysis.orchestrator import PostAnalysisOrchestrator


logger = logging.getLogger(__name__)

post_analysis_bp = Blueprint("post_analysis", __name__)
store = ArtifactStore()

JsonDict = Dict[str, Any]


def _should_use_celery() -> bool:
    """Decide whether to enqueue work on Celery or run synchronously.

    Rationale: local `python -m flask ...` runs often don't have Redis/Celery
    available, which caused recurring 503s and noisy retry logs. In Docker,
    broker URLs typically point at the `redis` service and Celery is expected.
    """

    override = os.getenv("POST_ANALYSIS_USE_CELERY")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no"}

    broker_url = os.getenv("CELERY_BROKER_URL", "")
    host = urlparse(broker_url).hostname
    # If broker is localhost, default to sync to avoid requiring Redis.
    return host not in {None, "localhost", "127.0.0.1", "::1"}


@post_analysis_bp.route("/post_analysis/generate", methods=["POST"])
def generate_post_analysis():
    """Trigger Analysis Overview report generation.

    Body (optional):
    {
      "run_id": "optional",
      "task_ids": {
        "enterprise": "<task_id>",
        "market": "<task_id>",
        "strategy": "<task_id>",
        "performance": "<task_id>",
        "optimization": "<task_id>",
        "shap": "<task_id>"  # best-effort
      }
    }
    """

    data: JsonDict = request.get_json(silent=True) or {}
    task_ids = data.get("task_ids")
    dataset = data.get("dataset")

    run_id = data.get("run_id") or new_run_id()

    # Create a placeholder status so the dashboard can show the run immediately.
    store.update_status(run_id, state="queued")
    store.set_latest_run(run_id)

    if _should_use_celery():
        try:
            task = generate_analysis_overview_async.delay(run_id=run_id, task_ids=task_ids, dataset=dataset)
            return jsonify({
                "run_id": run_id,
                "celery_task_id": task.id,
                "status_url": f"/api/v1/post_analysis/runs/{run_id}/status",
            }), 202
        except OperationalError as e:
            logger.warning(f"Celery unavailable; falling back to sync generation: {e}")

    # Synchronous fallback (no Redis/Celery required).
    try:
        orchestrator = PostAnalysisOrchestrator(store=store)
        result = orchestrator.run(run_id=run_id, task_ids=task_ids, dataset=dataset, include_audio=False)
        return jsonify({
            "run_id": run_id,
            "status_url": f"/api/v1/post_analysis/runs/{run_id}/status",
            "mode": "sync",
            "result": result,
        }), 200
    except Exception as e:
        logger.error(f"Synchronous post-analysis generation failed: {e}", exc_info=True)
        store.update_status(run_id, state="failed", error="generation_failed")
        return jsonify({
            "error": "generation_failed",
            "message": "Analysis Overview generation failed.",
            "details": str(e),
            "run_id": run_id,
        }), 500


@post_analysis_bp.route("/post_analysis/runs/<run_id>/status", methods=["GET"])
def get_post_analysis_status(run_id: str):
    status = store.get_status(run_id)
    if not status:
        return jsonify({"error": "Not Found", "message": "Unknown run_id", "status_code": 404}), 404
    return jsonify(status), 200


@post_analysis_bp.route("/post_analysis/runs/<run_id>/podcast", methods=["POST"])
def generate_podcast_for_run(run_id: str):
    """Generate a podcast summary MP3 from the run's final report text."""

    status = store.get_status(run_id)
    if not status:
        return jsonify({"error": "Not Found", "message": "Unknown run_id", "status_code": 404}), 404

    # Require a completed report first.
    paths = store.paths(run_id)
    if not paths.report_txt_path.exists():
        return jsonify({
            "error": "Report Not Ready",
            "message": "Generate the Analysis Overview report first.",
            "status_code": 409,
        }), 409

    if _should_use_celery():
        try:
            task = generate_podcast_summary_async.delay(run_id=run_id)
            return jsonify({
                "run_id": run_id,
                "celery_task_id": task.id,
                "status_url": f"/api/v1/post_analysis/runs/{run_id}/status",
                "artifact_url": f"/api/v1/post_analysis/runs/{run_id}/artifact/audio.mp3?download=1",
            }), 202
        except OperationalError as e:
            logger.warning(f"Celery unavailable; falling back to sync podcast generation: {e}")

    # Synchronous fallback: generate a podcast script from report text, then TTS.
    try:
        from src.post_analysis.audio import PodcastAudioGenerator
        from src.post_analysis.podcast_script import PodcastScriptGenerator, fallback_script

        store.update_status(run_id, state="generating_podcast")
        report_text = paths.report_txt_path.read_text(encoding="utf-8")

        try:
            script_text = PodcastScriptGenerator().generate(report_text)
            script_mode = "llm"
        except Exception as e:
            logger.warning(f"Podcast script LLM generation failed; using fallback script: {e}")
            script_text = fallback_script(report_text)
            script_mode = "fallback"

        paths.podcast_script_path.parent.mkdir(parents=True, exist_ok=True)
        paths.podcast_script_path.write_text(script_text, encoding="utf-8")

        audio_engine = PodcastAudioGenerator()
        audio_meta = audio_engine.synthesize_to_mp3(script_text, paths.audio_mp3_path)

        store.update_status(
            run_id,
            state="ready",
            podcast_script=str(paths.podcast_script_path),
            podcast_script_mode=script_mode,
            audio_mp3=str(paths.audio_mp3_path),
            audio_meta={"sample_rate": audio_meta.sample_rate, "bit_rate_kbps": audio_meta.bit_rate_kbps},
        )

        return jsonify({
            "run_id": run_id,
            "status_url": f"/api/v1/post_analysis/runs/{run_id}/status",
            "artifact_url": f"/api/v1/post_analysis/runs/{run_id}/artifact/audio.mp3?download=1",
            "mode": "sync",
        }), 200
    except Exception as e:
        logger.error(f"Synchronous podcast generation failed: {e}", exc_info=True)
        store.update_status(run_id, state="failed", error="podcast_generation_failed")
        return jsonify({
            "error": "podcast_generation_failed",
            "message": "Podcast generation failed.",
            "details": str(e),
            "run_id": run_id,
        }), 500


@post_analysis_bp.route("/post_analysis/runs/<run_id>/artifact/<artifact_name>", methods=["GET"])
def get_post_analysis_artifact(run_id: str, artifact_name: str):
    paths = store.paths(run_id)

    name_map = {
        "snapshot.json": paths.snapshot_path,
        "report.md": paths.report_md_path,
        "report.txt": paths.report_txt_path,
        "report.pdf": paths.report_pdf_path,
        "podcast_script.txt": paths.podcast_script_path,
        "audio.mp3": paths.audio_mp3_path,
        "status.json": paths.status_path,
    }

    path = name_map.get(artifact_name)
    if not path or not path.exists():
        return jsonify({"error": "Not Found", "message": "Artifact not found", "status_code": 404}), 404

    return send_file(path, as_attachment=("download" in request.args))
