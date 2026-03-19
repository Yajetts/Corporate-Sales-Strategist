from __future__ import annotations

import json
from types import MappingProxyType
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Mapping, Optional


JsonDict = Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {k: _deep_freeze(v) for k, v in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _deep_unfreeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _deep_unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_deep_unfreeze(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """Immutable snapshot of a completed analysis run.

    This is the only object the report/audio layers need. It is frozen so it represents
    a completed run and can't be mutated in-place.
    """

    enterprise_analysis: Optional[JsonDict] = None
    market_segments: Optional[JsonDict] = None
    strategy_recommendations: Optional[JsonDict] = None
    performance_forecasts: Optional[JsonDict] = None
    resource_allocations: Optional[JsonDict] = None
    shap_explanations: Optional[JsonDict] = None

    metadata: JsonDict = field(default_factory=dict)

    snapshot_id: str = field(default="", repr=True)

    def __post_init__(self):
        # Ensure metadata always has timestamps and a deterministic snapshot_id.
        md = dict(self.metadata or {})
        md.setdefault("created_at", _utc_now_iso())
        md.setdefault("schema_version", 1)

        # Hash should not depend on volatile timestamps.
        md_for_id = dict(md)
        md_for_id.pop("created_at", None)
        md_for_id.pop("updated_at", None)

        enterprise = _deep_freeze(self.enterprise_analysis) if self.enterprise_analysis is not None else None
        market = _deep_freeze(self.market_segments) if self.market_segments is not None else None
        strategy = _deep_freeze(self.strategy_recommendations) if self.strategy_recommendations is not None else None
        performance = _deep_freeze(self.performance_forecasts) if self.performance_forecasts is not None else None
        resources = _deep_freeze(self.resource_allocations) if self.resource_allocations is not None else None
        shap = _deep_freeze(self.shap_explanations) if self.shap_explanations is not None else None
        md_frozen = _deep_freeze(md)

        base_obj = {
            "enterprise_analysis": _deep_unfreeze(enterprise),
            "market_segments": _deep_unfreeze(market),
            "strategy_recommendations": _deep_unfreeze(strategy),
            "performance_forecasts": _deep_unfreeze(performance),
            "resource_allocations": _deep_unfreeze(resources),
            "shap_explanations": _deep_unfreeze(shap),
            "metadata": md_for_id,
        }
        digest = sha256(_stable_dumps(base_obj).encode("utf-8")).hexdigest()

        object.__setattr__(self, "enterprise_analysis", enterprise)
        object.__setattr__(self, "market_segments", market)
        object.__setattr__(self, "strategy_recommendations", strategy)
        object.__setattr__(self, "performance_forecasts", performance)
        object.__setattr__(self, "resource_allocations", resources)
        object.__setattr__(self, "shap_explanations", shap)
        object.__setattr__(self, "metadata", md_frozen)
        object.__setattr__(self, "snapshot_id", digest)

    def to_dict(self) -> JsonDict:
        return {
            "snapshot_id": self.snapshot_id,
            "enterprise_analysis": _deep_unfreeze(self.enterprise_analysis),
            "market_segments": _deep_unfreeze(self.market_segments),
            "strategy_recommendations": _deep_unfreeze(self.strategy_recommendations),
            "performance_forecasts": _deep_unfreeze(self.performance_forecasts),
            "resource_allocations": _deep_unfreeze(self.resource_allocations),
            "shap_explanations": _deep_unfreeze(self.shap_explanations),
            "metadata": _deep_unfreeze(self.metadata),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AnalysisSnapshot":
        # snapshot_id is recomputed in __post_init__ for integrity.
        return AnalysisSnapshot(
            enterprise_analysis=data.get("enterprise_analysis"),
            market_segments=data.get("market_segments"),
            strategy_recommendations=data.get("strategy_recommendations"),
            performance_forecasts=data.get("performance_forecasts"),
            resource_allocations=data.get("resource_allocations"),
            shap_explanations=data.get("shap_explanations"),
            metadata=dict(data.get("metadata") or {}),
        )
