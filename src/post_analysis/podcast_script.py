from __future__ import annotations

import logging
import os

from src.post_analysis.llm_client import LLMTextClient
from src.post_analysis.prompts.podcast_prompt import build_podcast_prompt


logger = logging.getLogger(__name__)


class PodcastScriptGenerator:
    """Generate a podcast-style narration script from the final report text.

    This does not re-run any modules. It only uses the already-generated report.
    """

    def generate(self, report_text: str) -> str:
        style = os.getenv("POST_ANALYSIS_PODCAST_STYLE", "two_host")
        target_minutes = int(os.getenv("POST_ANALYSIS_PODCAST_TARGET_MINUTES", "6"))
        include_disclaimer = os.getenv("POST_ANALYSIS_PODCAST_DISCLAIMER", "1").strip().lower() not in {"0", "false", "no"}

        prompt = build_podcast_prompt(
            report_text=report_text,
            style=style,
            target_minutes=target_minutes,
            include_disclaimer=include_disclaimer,
        )

        client = LLMTextClient()
        script = client.generate_markdown(prompt)
        return (script or "").strip() + "\n"


def _slice_between(report: str, start: str, end_markers: list[str]) -> str:
    if not report:
        return ""
    if start not in report:
        return ""
    after = report.split(start, 1)[1]
    end_idx = None
    for m in end_markers:
        i = after.find(m)
        if i != -1:
            end_idx = i if end_idx is None else min(end_idx, i)
    chunk = after if end_idx is None else after[:end_idx]
    return chunk.strip()


def _compact_lines(block: str, *, max_lines: int) -> list[str]:
    lines = [ln.strip() for ln in (block or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return lines[:max_lines]


def fallback_script(report_text: str) -> str:
    """Best-effort fallback when no LLM is configured.

    Produces something more spoken than reading the full report, but still grounded.
    """

    # More structured, module-by-module spoken summary.
    intro = (
        "Let's walk through your Analysis Overview report together. "
        "I'll summarize each module's output and explain what your report is concluding in plain language.\n"
    )

    # Headings are present in the markdown report; if we only have plain text, this still works
    # because headings appear as lines like "1. Executive Summary" etc.
    markers = {
        "exec": ["## 1. Executive Summary", "1. Executive Summary"],
        "company": ["## 2. Company & Product Understanding", "2. Company & Product Understanding"],
        "market": ["## 3. Market Insights & Customer Segmentation", "3. Market Insights & Customer Segmentation"],
        "strategy": ["## 4. Sales & Pricing Strategy Recommendations", "4. Sales & Pricing Strategy Recommendations"],
        "perf": ["## 5. Performance Outlook & Risk Assessment", "5. Performance Outlook & Risk Assessment"],
        "ops": ["## 6. Operational & Resource Optimization", "6. Operational & Resource Optimization"],
        "shap": ["## 7. Explainability & Model Confidence", "7. Explainability & Model Confidence"],
        "takeaways": ["## 8. Final Takeaways", "8. Final Takeaways"],
    }

    end_markers = [
        "## 1. Executive Summary",
        "## 2. Company & Product Understanding",
        "## 3. Market Insights & Customer Segmentation",
        "## 4. Sales & Pricing Strategy Recommendations",
        "## 5. Performance Outlook & Risk Assessment",
        "## 6. Operational & Resource Optimization",
        "## 7. Explainability & Model Confidence",
        "## 8. Final Takeaways",
        "End of Analysis Overview",
    ]

    def _first_marker(keys: list[str]) -> str:
        for k in keys:
            if k in report_text:
                return k
        # fallback to first
        return keys[0]

    def _section(keys: list[str]) -> str:
        start = _first_marker(keys)
        return _slice_between(report_text, start, end_markers)

    parts: list[str] = [intro]

    exec_block = _section(markers["exec"])
    if exec_block:
        lines = _compact_lines(exec_block, max_lines=5)
        parts.append("Executive summary: " + " ".join(lines[:3]) + "\n")

    company_block = _section(markers["company"])
    if company_block:
        lines = _compact_lines(company_block, max_lines=8)
        parts.append("Company and product understanding: " + " ".join(lines[:4]) + "\n")

    market_block = _section(markers["market"])
    if market_block:
        lines = _compact_lines(market_block, max_lines=12)
        parts.append("Market insights and customer segmentation: " + " ".join(lines[:6]) + "\n")

    strategy_block = _section(markers["strategy"])
    if strategy_block:
        lines = _compact_lines(strategy_block, max_lines=12)
        parts.append("Sales and pricing strategy: " + " ".join(lines[:6]) + "\n")

    perf_block = _section(markers["perf"])
    if perf_block:
        lines = _compact_lines(perf_block, max_lines=14)
        parts.append("Performance outlook and risks: " + " ".join(lines[:7]) + "\n")

    ops_block = _section(markers["ops"])
    if ops_block:
        lines = _compact_lines(ops_block, max_lines=12)
        parts.append("Operational and resource optimization: " + " ".join(lines[:6]) + "\n")

    shap_block = _section(markers["shap"])
    if shap_block:
        lines = _compact_lines(shap_block, max_lines=12)
        parts.append("Explainability and model confidence: " + " ".join(lines[:6]) + "\n")

    takeaways_block = _section(markers["takeaways"])
    if takeaways_block:
        lines = _compact_lines(takeaways_block, max_lines=14)
        parts.append("Final takeaways: " + " ".join(lines[:8]) + "\n")

    parts.append(
        "That wraps up the module-by-module overview. Next, take the recommendations in your report, "
        "track the performance signals, and rerun any missing modules to increase confidence.\n"
    )

    if os.getenv("POST_ANALYSIS_PODCAST_DISCLAIMER", "1").strip().lower() not in {"0", "false", "no"}:
        parts.append("Quick note: this is an automated spoken summary of your report and may omit details.\n")

    script = "\n".join(p.strip() for p in parts if p and p.strip())
    max_chars = int(os.getenv("POST_ANALYSIS_PODCAST_FALLBACK_MAX_CHARS", "7000"))
    if len(script) > max_chars:
        script = script[:max_chars].rstrip() + "\n\nAdditional details are available in the full PDF report."
    return script.strip() + "\n"
