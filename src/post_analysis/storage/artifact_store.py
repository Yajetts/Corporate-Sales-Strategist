from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


JsonDict = Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> Path:
    # src/post_analysis/storage/artifact_store.py -> storage -> post_analysis -> src -> repo root
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    run_id: str
    root_dir: Path

    @property
    def run_dir(self) -> Path:
        return self.root_dir / self.run_id

    @property
    def snapshot_path(self) -> Path:
        return self.run_dir / "snapshot.json"

    @property
    def report_md_path(self) -> Path:
        return self.run_dir / "report.md"

    @property
    def report_txt_path(self) -> Path:
        return self.run_dir / "report.txt"

    @property
    def report_pdf_path(self) -> Path:
        return self.run_dir / "report.pdf"

    @property
    def audio_mp3_path(self) -> Path:
        return self.run_dir / "audio.mp3"

    @property
    def podcast_script_path(self) -> Path:
        return self.run_dir / "podcast_script.txt"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"


class ArtifactStore:
    """File-based artifact store (loosely coupled, no DB schema changes)."""

    def __init__(self, root_dir: Optional[str] = None):
        root_dir = root_dir or os.getenv("POST_ANALYSIS_DIR", "logs/post_analysis")
        root_path = Path(root_dir)
        if not root_path.is_absolute():
            root_path = _project_root() / root_path
        self.root_dir = root_path
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.root_dir / "latest.json"

    def paths(self, run_id: str) -> ArtifactPaths:
        return ArtifactPaths(run_id=run_id, root_dir=self.root_dir)

    def ensure_run_dir(self, run_id: str) -> Path:
        p = self.paths(run_id).run_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, path: Path, data: JsonDict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def read_json(self, path: Path) -> Optional[JsonDict]:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def update_status(self, run_id: str, **fields: Any) -> JsonDict:
        paths = self.paths(run_id)
        self.ensure_run_dir(run_id)
        current = self.read_json(paths.status_path) or {}
        current.update(fields)
        current.setdefault("run_id", run_id)
        current.setdefault("updated_at", _utc_now_iso())
        current["updated_at"] = _utc_now_iso()
        self.write_json(paths.status_path, current)
        return current

    def set_latest_run(self, run_id: str) -> None:
        self.write_json(self.latest_path, {"run_id": run_id, "updated_at": _utc_now_iso()})

    def get_latest_run(self) -> Optional[str]:
        data = self.read_json(self.latest_path) or {}
        run_id = data.get("run_id")
        return str(run_id) if run_id else None

    def get_status(self, run_id: str) -> Optional[JsonDict]:
        return self.read_json(self.paths(run_id).status_path)
