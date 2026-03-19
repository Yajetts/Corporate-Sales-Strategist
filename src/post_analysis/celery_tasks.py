from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.api.celery_app import celery, BaseTask
from src.post_analysis.orchestrator import PostAnalysisOrchestrator
from src.post_analysis.audio import PodcastAudioGenerator
from src.post_analysis.storage.artifact_store import ArtifactStore


logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


@celery.task(base=BaseTask, bind=True, name="src.post_analysis.celery_tasks.generate_analysis_overview_async")
def generate_analysis_overview_async(
    self,
    run_id: Optional[str] = None,
    task_ids: Optional[JsonDict] = None,
    dataset: Optional[str] = None,
):
    """Generate unified post-analysis report as a Celery task."""

    try:
        self.update_state(state="STARTED", meta={"step": "starting"})
        orchestrator = PostAnalysisOrchestrator()

        self.update_state(state="PROGRESS", meta={"step": "collecting"})
        result = orchestrator.run(run_id=run_id, task_ids=task_ids, dataset=dataset, include_audio=False)

        self.update_state(state="SUCCESS", meta={"step": "ready"})
        return result
    except Exception as e:
        logger.error(f"generate_analysis_overview_async failed: {e}", exc_info=True)
        raise


@celery.task(base=BaseTask, bind=True, name="src.post_analysis.celery_tasks.generate_podcast_summary_async")
def generate_podcast_summary_async(self, run_id: str):
    """Generate a podcast-style MP3 from the already-generated report text."""

    store = ArtifactStore()
    paths = store.paths(run_id)
    if not paths.report_txt_path.exists():
        raise FileNotFoundError("Report text not found for run_id")

    try:
        store.update_status(run_id, state="generating_podcast")
        self.update_state(state="PROGRESS", meta={"step": "generating_podcast"})

        report_text = paths.report_txt_path.read_text(encoding="utf-8")

        from src.post_analysis.podcast_script import PodcastScriptGenerator, fallback_script

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

        self.update_state(state="SUCCESS", meta={"step": "ready"})
        return {"run_id": run_id, "status": store.get_status(run_id)}
    except Exception as e:
        logger.error(f"generate_podcast_summary_async failed: {e}", exc_info=True)
        store.update_status(run_id, state="failed", error="podcast_generation_failed")
        raise
