from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


JsonDict = Dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    markdown: str
    text: str
    module_contributions: JsonDict
    run_id: str
    snapshot_id: str
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
