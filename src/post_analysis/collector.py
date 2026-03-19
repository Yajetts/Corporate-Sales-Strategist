from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import desc

from src.api.database import (
    postgres_manager,
    mongodb_manager,
    AnalysisResult,
    MarketAnalysisResult,
    StrategyResult,
    PerformanceResult,
    BusinessOptimizationResult,
)
from src.post_analysis.snapshot import AnalysisSnapshot
from src.post_analysis.storage.module_output_store import ModuleOutputStore


logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModuleTaskIds:
    enterprise_task_id: Optional[str] = None
    market_task_id: Optional[str] = None
    strategy_task_id: Optional[str] = None
    performance_task_id: Optional[str] = None
    optimization_task_id: Optional[str] = None
    shap_task_id: Optional[str] = None  # best-effort (Celery-only unless persisted elsewhere)


class UnifiedOutputCollector:
    """Best-effort collector that reads existing module outputs without modifying them."""

    def collect(self, task_ids: Optional[ModuleTaskIds] = None) -> AnalysisSnapshot:
        task_ids = task_ids or ModuleTaskIds()

        store = ModuleOutputStore()

        enterprise, enterprise_meta = self._get_enterprise_analysis(task_ids.enterprise_task_id, store=store)
        market, market_meta = self._get_market_analysis(task_ids.market_task_id, store=store)
        strategy, strategy_meta = self._get_strategy(task_ids.strategy_task_id, store=store)
        performance, performance_meta = self._get_performance(task_ids.performance_task_id, store=store)
        optimization, optimization_meta = self._get_optimization(task_ids.optimization_task_id, store=store)

        # SHAP explainability: best-effort from local cache (sync routes / async task cache).
        shap, shap_meta = self._get_shap(task_ids.shap_task_id, store=store)

        # Derive a stable, report-friendly "segments" view for Market Decipherer.
        market = _with_derived_market_segments(market)

        metadata: JsonDict = {
            "module_task_ids": {
                "enterprise": enterprise_meta.get("task_id"),
                "market": market_meta.get("task_id"),
                "strategy": strategy_meta.get("task_id"),
                "performance": performance_meta.get("task_id"),
                "optimization": optimization_meta.get("task_id"),
                "shap": shap_meta.get("task_id") or task_ids.shap_task_id,
            },
            "collector": {
                "notes": "SHAP is best-effort and collected from local cache unless persisted elsewhere.",
                "sources": {
                    "enterprise": enterprise_meta.get("source"),
                    "market": market_meta.get("source"),
                    "strategy": strategy_meta.get("source"),
                    "performance": performance_meta.get("source"),
                    "optimization": optimization_meta.get("source"),
                    "shap": shap_meta.get("source"),
                },
            },
        }

        return AnalysisSnapshot(
            enterprise_analysis=enterprise,
            market_segments=market,
            strategy_recommendations=strategy,
            performance_forecasts=performance,
            resource_allocations=optimization,
            shap_explanations=shap,
            metadata=metadata,
        )

    def _get_shap(self, task_id: Optional[str], *, store: ModuleOutputStore) -> Tuple[Optional[JsonDict], JsonDict]:
        """Collect SHAP explainability output from the local cache.

        The core API does not persist SHAP outputs in Postgres/Mongo today, so we treat
        this as a best-effort artifact primarily for the Analysis Overview report.
        """

        latest = store.read_latest("shap")
        if not latest:
            return None, {"task_id": task_id, "source": "none"}

        if task_id and latest.task_id != task_id:
            # We only keep the latest SHAP; if the requested task_id doesn't match,
            # don't accidentally show a different run's explanation.
            return None, {"task_id": task_id, "source": "none"}

        payload = dict(latest.payload)
        payload.setdefault("created_at", latest.completed_at)
        return payload, {"task_id": latest.task_id, "source": "local_cache"}


    def _get_enterprise_analysis(self, task_id: Optional[str], *, store: ModuleOutputStore) -> Tuple[Optional[JsonDict], JsonDict]:
        if not task_id:
            latest = store.read_latest("enterprise")
            if latest:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            try:
                with postgres_manager.get_session() as session:
                    row = session.query(AnalysisResult).order_by(desc(AnalysisResult.created_at)).first()
                    if not row:
                        return None, {"task_id": None, "source": "none"}
                    return {
                        "task_id": row.task_id,
                        "product_category": row.product_category,
                        "business_domain": row.business_domain,
                        "value_proposition": row.value_proposition,
                        "key_features": row.key_features,
                        "confidence_scores": row.confidence_scores,
                        "processing_time_ms": row.processing_time_ms,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }, {"task_id": row.task_id, "source": "db_latest"}
            except Exception as e:
                logger.warning(f"Failed to collect latest enterprise analysis: {e}")
                return None, {"task_id": None, "source": "none"}
        try:
            with postgres_manager.get_session() as session:
                q = session.query(AnalysisResult)
                row = q.filter_by(task_id=task_id).first()

                if not row:
                    latest = store.read_latest("enterprise")
                    if latest and latest.task_id == task_id:
                        payload = dict(latest.payload)
                        payload.setdefault("created_at", latest.completed_at)
                        return payload, {"task_id": latest.task_id, "source": "local_cache"}
                    return None, {"task_id": task_id, "source": "none"}

                return {
                    "task_id": row.task_id,
                    "product_category": row.product_category,
                    "business_domain": row.business_domain,
                    "value_proposition": row.value_proposition,
                    "key_features": row.key_features,
                    "confidence_scores": row.confidence_scores,
                    "processing_time_ms": row.processing_time_ms,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }, {"task_id": row.task_id, "source": "db_by_task_id"}
        except Exception as e:
            logger.warning(f"Failed to collect enterprise analysis: {e}")
            latest = store.read_latest("enterprise")
            if latest and latest.task_id == task_id:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            return None, {"task_id": task_id, "source": "none"}

    def _get_market_analysis(self, task_id: Optional[str], *, store: ModuleOutputStore) -> Tuple[Optional[JsonDict], JsonDict]:
        if not task_id:
            latest = store.read_latest("market")
            if latest:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            try:
                with postgres_manager.get_session() as session:
                    row = session.query(MarketAnalysisResult).order_by(desc(MarketAnalysisResult.created_at)).first()
                    if not row:
                        return None, {"task_id": None, "source": "none"}
                    details = mongodb_manager.get_collection("market_analysis_details").find_one({"task_id": row.task_id})
                    return {
                        "task_id": row.task_id,
                        "num_entities": row.num_entities,
                        "num_clusters": row.num_clusters,
                        "clustering_method": row.clustering_method,
                        "latent_dimensions": row.latent_dimensions,
                        "processing_time_seconds": row.processing_time_seconds,
                        "clusters": (details or {}).get("clusters"),
                        "graph": (details or {}).get("graph"),
                        "potential_clients": (details or {}).get("potential_clients"),
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }, {"task_id": row.task_id, "source": "db_latest"}
            except Exception as e:
                logger.warning(f"Failed to collect latest market analysis: {e}")
                return None, {"task_id": None, "source": "none"}
        try:
            with postgres_manager.get_session() as session:
                q = session.query(MarketAnalysisResult)
                row = q.filter_by(task_id=task_id).first()

                if not row:
                    latest = store.read_latest("market")
                    if latest and latest.task_id == task_id:
                        payload = dict(latest.payload)
                        payload.setdefault("created_at", latest.completed_at)
                        return payload, {"task_id": latest.task_id, "source": "local_cache"}
                    return None, {"task_id": task_id, "source": "none"}

                details = mongodb_manager.get_collection("market_analysis_details").find_one({"task_id": row.task_id})

                return {
                    "task_id": row.task_id,
                    "num_entities": row.num_entities,
                    "num_clusters": row.num_clusters,
                    "clustering_method": row.clustering_method,
                    "latent_dimensions": row.latent_dimensions,
                    "processing_time_seconds": row.processing_time_seconds,
                    "clusters": (details or {}).get("clusters"),
                    "graph": (details or {}).get("graph"),
                    "potential_clients": (details or {}).get("potential_clients"),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }, {"task_id": row.task_id, "source": "db_by_task_id"}
        except Exception as e:
            logger.warning(f"Failed to collect market analysis: {e}")
            latest = store.read_latest("market")
            if latest and latest.task_id == task_id:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            return None, {"task_id": task_id, "source": "none"}

    def _get_strategy(self, task_id: Optional[str], *, store: ModuleOutputStore) -> Tuple[Optional[JsonDict], JsonDict]:
        if not task_id:
            latest = store.read_latest("strategy")
            if latest:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            try:
                with postgres_manager.get_session() as session:
                    row = session.query(StrategyResult).order_by(desc(StrategyResult.created_at)).first()
                    if not row:
                        return None, {"task_id": None, "source": "none"}
                    explanation_doc = None
                    if row.has_explanation:
                        explanation_doc = mongodb_manager.get_collection("strategy_explanations").find_one({"task_id": row.task_id})
                    return {
                        "task_id": row.task_id,
                        "market_state": row.market_state,
                        "recommendations": row.recommendations,
                        "confidence_score": row.confidence_score,
                        "confidence_level": row.confidence_level,
                        "explanation": (explanation_doc or {}).get("explanation"),
                        "actionable_insights": (explanation_doc or {}).get("actionable_insights"),
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }, {"task_id": row.task_id, "source": "db_latest"}
            except Exception as e:
                logger.warning(f"Failed to collect latest strategy result: {e}")
                return None, {"task_id": None, "source": "none"}
        try:
            with postgres_manager.get_session() as session:
                q = session.query(StrategyResult)
                row = q.filter_by(task_id=task_id).first()

                if not row:
                    latest = store.read_latest("strategy")
                    if latest and latest.task_id == task_id:
                        payload = dict(latest.payload)
                        payload.setdefault("created_at", latest.completed_at)
                        return payload, {"task_id": latest.task_id, "source": "local_cache"}
                    return None, {"task_id": task_id, "source": "none"}

                explanation_doc = None
                if row.has_explanation:
                    explanation_doc = mongodb_manager.get_collection("strategy_explanations").find_one({"task_id": row.task_id})

                return {
                    "task_id": row.task_id,
                    "market_state": row.market_state,
                    "recommendations": row.recommendations,
                    "confidence_score": row.confidence_score,
                    "confidence_level": row.confidence_level,
                    "explanation": (explanation_doc or {}).get("explanation"),
                    "actionable_insights": (explanation_doc or {}).get("actionable_insights"),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }, {"task_id": row.task_id, "source": "db_by_task_id"}
        except Exception as e:
            logger.warning(f"Failed to collect strategy result: {e}")
            latest = store.read_latest("strategy")
            if latest and latest.task_id == task_id:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            return None, {"task_id": task_id, "source": "none"}

    def _get_performance(self, task_id: Optional[str], *, store: ModuleOutputStore) -> Tuple[Optional[JsonDict], JsonDict]:
        if not task_id:
            latest = store.read_latest("performance")
            if latest:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            try:
                with postgres_manager.get_session() as session:
                    row = session.query(PerformanceResult).order_by(desc(PerformanceResult.created_at)).first()
                    if not row:
                        return None, {"task_id": None, "source": "none"}
                    details = mongodb_manager.get_collection("performance_details").find_one({"task_id": row.task_id})
                    return {
                        "task_id": row.task_id,
                        "forecast_horizon_days": row.forecast_horizon_days,
                        "num_alerts": row.num_alerts,
                        "critical_alerts": row.critical_alerts,
                        "processing_time_seconds": row.processing_time_seconds,
                        "forecast": (details or {}).get("forecast"),
                        "confidence_intervals": (details or {}).get("confidence_intervals"),
                        "alerts": (details or {}).get("alerts"),
                        "feedback_summary": (details or {}).get("feedback_summary"),
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }, {"task_id": row.task_id, "source": "db_latest"}
            except Exception as e:
                logger.warning(f"Failed to collect latest performance result: {e}")
                return None, {"task_id": None, "source": "none"}
        try:
            with postgres_manager.get_session() as session:
                q = session.query(PerformanceResult)
                row = q.filter_by(task_id=task_id).first()

                if not row:
                    latest = store.read_latest("performance")
                    if latest and latest.task_id == task_id:
                        payload = dict(latest.payload)
                        payload.setdefault("created_at", latest.completed_at)
                        return payload, {"task_id": latest.task_id, "source": "local_cache"}
                    return None, {"task_id": task_id, "source": "none"}

                details = mongodb_manager.get_collection("performance_details").find_one({"task_id": row.task_id})

                return {
                    "task_id": row.task_id,
                    "forecast_horizon_days": row.forecast_horizon_days,
                    "num_alerts": row.num_alerts,
                    "critical_alerts": row.critical_alerts,
                    "processing_time_seconds": row.processing_time_seconds,
                    "forecast": (details or {}).get("forecast"),
                    "confidence_intervals": (details or {}).get("confidence_intervals"),
                    "alerts": (details or {}).get("alerts"),
                    "feedback_summary": (details or {}).get("feedback_summary"),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }, {"task_id": row.task_id, "source": "db_by_task_id"}
        except Exception as e:
            logger.warning(f"Failed to collect performance result: {e}")
            latest = store.read_latest("performance")
            if latest and latest.task_id == task_id:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            return None, {"task_id": task_id, "source": "none"}

    def _get_optimization(self, task_id: Optional[str], *, store: ModuleOutputStore) -> Tuple[Optional[JsonDict], JsonDict]:
        if not task_id:
            latest = store.read_latest("optimization")
            if latest:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            try:
                with postgres_manager.get_session() as session:
                    row = session.query(BusinessOptimizationResult).order_by(desc(BusinessOptimizationResult.created_at)).first()
                    if not row:
                        return None, {"task_id": None, "source": "none"}
                    details = mongodb_manager.get_collection("business_optimization_details").find_one({"task_id": row.task_id})
                    return {
                        "task_id": row.task_id,
                        "num_products": row.num_products,
                        "total_revenue": row.total_revenue,
                        "total_cost": row.total_cost,
                        "profit": row.profit,
                        "roi": row.roi,
                        "optimization_success": bool(row.optimization_success),
                        "processing_time_seconds": row.processing_time_seconds,
                        "production_priorities": (details or {}).get("production_priorities"),
                        "resource_distribution": (details or {}).get("resource_distribution"),
                        "constraints_applied": (details or {}).get("constraints_applied"),
                        "optimization_details": (details or {}).get("optimization_details"),
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }, {"task_id": row.task_id, "source": "db_latest"}
            except Exception as e:
                logger.warning(f"Failed to collect latest business optimization result: {e}")
                return None, {"task_id": None, "source": "none"}
        try:
            with postgres_manager.get_session() as session:
                q = session.query(BusinessOptimizationResult)
                row = q.filter_by(task_id=task_id).first()

                if not row:
                    latest = store.read_latest("optimization")
                    if latest and latest.task_id == task_id:
                        payload = dict(latest.payload)
                        payload.setdefault("created_at", latest.completed_at)
                        return payload, {"task_id": latest.task_id, "source": "local_cache"}
                    return None, {"task_id": task_id, "source": "none"}

                details = mongodb_manager.get_collection("business_optimization_details").find_one({"task_id": row.task_id})

                return {
                    "task_id": row.task_id,
                    "num_products": row.num_products,
                    "total_revenue": row.total_revenue,
                    "total_cost": row.total_cost,
                    "profit": row.profit,
                    "roi": row.roi,
                    "optimization_success": bool(row.optimization_success),
                    "processing_time_seconds": row.processing_time_seconds,
                    "production_priorities": (details or {}).get("production_priorities"),
                    "resource_distribution": (details or {}).get("resource_distribution"),
                    "constraints_applied": (details or {}).get("constraints_applied"),
                    "optimization_details": (details or {}).get("optimization_details"),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }, {"task_id": row.task_id, "source": "db_by_task_id"}
        except Exception as e:
            logger.warning(f"Failed to collect business optimization result: {e}")
            latest = store.read_latest("optimization")
            if latest and latest.task_id == task_id:
                payload = dict(latest.payload)
                payload.setdefault("created_at", latest.completed_at)
                return payload, {"task_id": latest.task_id, "source": "local_cache"}
            return None, {"task_id": task_id, "source": "none"}


def _with_derived_market_segments(market: Optional[JsonDict]) -> Optional[JsonDict]:
    """Add a lightweight `segments` list derived from cluster profiles.

    Report generation expects a human-readable segments view. The Market Decipherer
    output shape is `clusters.profiles` (a dict keyed by cluster_id), so we normalize
    it to a list under `segments`.
    """

    if not isinstance(market, dict):
        return market

    if isinstance(market.get("segments"), list) and market.get("segments"):
        return market

    clusters = market.get("clusters")
    if not isinstance(clusters, dict):
        return market

    profiles = clusters.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        return market

    derived = []
    for _cid, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        cid = profile.get("cluster_id", _cid)
        size = profile.get("size")
        pct = profile.get("percentage")
        name = f"Cluster {cid}" if cid is not None else "Cluster"
        if isinstance(pct, (int, float)):
            name = f"{name} ({pct:.1f}%)"

        traits = []
        if isinstance(size, int):
            traits.append(f"Size: {size} entities")

        # Feature stats are latent_*; still useful to show as a proxy summary.
        feat_stats = profile.get("feature_stats")
        if isinstance(feat_stats, dict) and feat_stats:
            ranked = []
            for fname, stats in feat_stats.items():
                if not isinstance(stats, dict):
                    continue
                mean = stats.get("mean")
                if isinstance(mean, (int, float)):
                    ranked.append((abs(float(mean)), fname, float(mean)))
            ranked.sort(reverse=True)
            for _, fname, mean in ranked[:2]:
                traits.append(f"Signal: {fname} mean={mean:.3f}")

        derived.append({"name": name, "traits": traits})

    if derived:
        market = dict(market)
        market["segments"] = derived
    return market
