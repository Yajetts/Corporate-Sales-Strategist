from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any, Dict, Optional

from src.post_analysis.collector import UnifiedOutputCollector, ModuleTaskIds
from src.post_analysis.report_engine import ReportGenerationEngine
from src.post_analysis.audio import PodcastAudioGenerator
from src.post_analysis.run_id import new_run_id
from src.post_analysis.snapshot import AnalysisSnapshot
from src.post_analysis.storage.artifact_store import ArtifactStore
from src.post_analysis.pdf import write_simple_pdf


logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


class PostAnalysisOrchestrator:
    def __init__(self, store: Optional[ArtifactStore] = None):
        self.store = store or ArtifactStore()
        self.collector = UnifiedOutputCollector()
        self.report_engine = ReportGenerationEngine()
        self.audio_engine = PodcastAudioGenerator()

    def run(
        self,
        run_id: Optional[str] = None,
        task_ids: Optional[JsonDict] = None,
        dataset: Optional[str] = None,
        include_audio: bool = False,
    ) -> JsonDict:
        run_id = run_id or new_run_id()
        paths = self.store.paths(run_id)
        self.store.ensure_run_dir(run_id)

        module_task_ids = ModuleTaskIds(
            enterprise_task_id=(task_ids or {}).get("enterprise"),
            market_task_id=(task_ids or {}).get("market"),
            strategy_task_id=(task_ids or {}).get("strategy"),
            performance_task_id=(task_ids or {}).get("performance"),
            optimization_task_id=(task_ids or {}).get("optimization"),
            shap_task_id=(task_ids or {}).get("shap"),
        )

        self.store.update_status(run_id, state="collecting", include_audio=include_audio)
        snapshot: AnalysisSnapshot = self.collector.collect(module_task_ids)
        snapshot_dict = snapshot.to_dict()

        # Ensure the report has a stable, user-facing identifier.
        snapshot_dict.setdefault("metadata", {})
        snapshot_dict["metadata"]["run_id"] = run_id
        if "dataset" not in snapshot_dict["metadata"]:
            snapshot_dict["metadata"]["dataset"] = (
                dataset
                or os.getenv("POST_ANALYSIS_DATASET_NAME")
                or "Not provided"
            )

        snapshot = AnalysisSnapshot.from_dict(snapshot_dict)
        snapshot_dict = snapshot.to_dict()
        self.store.write_json(paths.snapshot_path, snapshot_dict)

        effective_task_ids = (snapshot_dict.get("metadata") or {}).get("module_task_ids") or {}

        def _non_empty(v: Any) -> bool:
            if v is None:
                return False
            if isinstance(v, str):
                return bool(v.strip())
            if isinstance(v, (list, tuple, set, dict)):
                return len(v) > 0
            return True

        module_execution = {
            "Enterprise Analyst": {
                "task_id": effective_task_ids.get("enterprise") or module_task_ids.enterprise_task_id,
                "has_output": _non_empty(snapshot.enterprise_analysis),
            },
            "Market Decipherer": {
                "task_id": effective_task_ids.get("market") or module_task_ids.market_task_id,
                "has_output": _non_empty(snapshot.market_segments),
            },
            "Strategy Engine": {
                "task_id": effective_task_ids.get("strategy") or module_task_ids.strategy_task_id,
                "has_output": _non_empty(snapshot.strategy_recommendations),
            },
            "Performance Governor": {
                "task_id": effective_task_ids.get("performance") or module_task_ids.performance_task_id,
                "has_output": _non_empty(snapshot.performance_forecasts),
            },
            "Business Manager": {
                "task_id": effective_task_ids.get("optimization") or module_task_ids.optimization_task_id,
                "has_output": _non_empty(snapshot.resource_allocations),
            },
            "Model Transparency (SHAP)": {
                "task_id": effective_task_ids.get("shap") or module_task_ids.shap_task_id,
                "has_output": _non_empty(snapshot.shap_explanations),
            },
        }

        # Full-run gating requires all modules, including SHAP.
        full_run_complete = all(info.get("has_output") for info in module_execution.values())

        self.store.update_status(run_id, state="generating_report", snapshot_id=snapshot.snapshot_id)
        artifacts = self.report_engine.generate(snapshot)

        self.store.update_status(
            run_id,
            state="writing_report",
            llm_provider=artifacts.llm_provider,
            llm_model=artifacts.llm_model,
        )
        paths.report_md_path.write_text(artifacts.markdown, encoding="utf-8")
        paths.report_txt_path.write_text(artifacts.text, encoding="utf-8")

        # Optional PDF
        try:
            write_simple_pdf(artifacts.text, paths.report_pdf_path, title=f"Analysis Overview ({run_id})")
        except Exception as e:
            logger.warning(f"PDF export unavailable: {e}")

        audio_meta = None
        if include_audio:
            self.store.update_status(run_id, state="generating_audio")
            audio_meta = self.audio_engine.synthesize_to_mp3(artifacts.text, paths.audio_mp3_path)

        self.store.update_status(
            run_id,
            state="ready",
            report_md=str(paths.report_md_path),
            report_txt=str(paths.report_txt_path),
            report_pdf=str(paths.report_pdf_path) if paths.report_pdf_path.exists() else None,
            audio_mp3=str(paths.audio_mp3_path) if paths.audio_mp3_path.exists() else None,
            audio_meta=(asdict(audio_meta) if audio_meta else None),
            module_contributions=artifacts.module_contributions,
            module_task_ids=(snapshot_dict.get("metadata") or {}).get("module_task_ids"),
            module_execution=module_execution,
            full_analysis_run_complete=bool(full_run_complete),
        )

        # Mark this as the latest completed post-analysis run.
        self.store.set_latest_run(run_id)

        return {
            "run_id": run_id,
            "snapshot_id": snapshot.snapshot_id,
            "status": self.store.get_status(run_id),
        }
