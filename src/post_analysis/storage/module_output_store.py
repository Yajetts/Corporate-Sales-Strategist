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
    # src/post_analysis/storage/module_output_store.py -> storage -> post_analysis -> src -> repo root
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class ModuleOutputRecord:
    module: str
    task_id: str
    completed_at: str
    payload: JsonDict


class ModuleOutputStore:
    """File-based store for the latest sync module outputs.

    This is a local-dev safety net when Postgres/Mongo are unavailable.
    It is intentionally simple: one JSON file per module.
    """

    def __init__(self, root_dir: Optional[str] = None):
        root_dir = root_dir or os.getenv("MODULE_OUTPUTS_DIR", "logs/module_outputs/latest")
        root_path = Path(root_dir)
        if not root_path.is_absolute():
            root_path = _project_root() / root_path
        self.root_dir = root_path
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, module: str) -> Path:
        safe = "".join(c for c in module.lower() if c.isalnum() or c in {"_", "-"}).strip("-")
        return self.root_dir / f"{safe}.json"

    def write_latest(self, module: str, *, task_id: str, payload: JsonDict) -> Path:
        record = {
            "module": module,
            "task_id": task_id,
            "completed_at": _utc_now_iso(),
            "payload": payload,
        }
        path = self._path_for(module)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def read_latest(self, module: str) -> Optional[ModuleOutputRecord]:
        path = self._path_for(module)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        task_id = data.get("task_id")
        payload = data.get("payload")
        completed_at = data.get("completed_at")
        if not task_id or not isinstance(payload, dict):
            return None
        return ModuleOutputRecord(module=module, task_id=str(task_id), completed_at=str(completed_at or ""), payload=payload)
