from __future__ import annotations

from typing import Optional


def build_podcast_prompt(
    *,
    report_text: str,
    style: str = "two_host",
    target_minutes: int = 6,
    include_disclaimer: bool = True,
) -> str:
    """Build a podcast-script prompt from the final report text.

    The script must:
    - Be strictly grounded in report_text (no hallucinations).
    - Sound like a natural spoken narration.
    - Prefer concrete numbers and specifics that appear in the report.

        style:
            - "single_host": one narrator
            - "two_host": (deprecated) treated as single narrator
    """

    style = (style or "two_host").strip().lower()
    if style not in {"single_host", "two_host"}:
        style = "two_host"

    minutes = max(2, min(int(target_minutes or 6), 20))

    roles = "Use a single narrator voice. Do NOT include speaker labels (no 'Host:', no 'Co-host:')."

    disclaimer = (
        "Include a short disclaimer that this is an automated summary of the report and may omit details."
        if include_disclaimer
        else ""
    )

    return f"""You are a business-professional spoken-summary script writer.

GOAL:
Write a spoken script that explains the provided report directly to the user.
The output should sound natural, confident, and business-professional (like an executive briefing), as if you're walking the user through what their report concludes.

CRITICAL CONSTRAINTS:
- Do NOT hallucinate. Use ONLY the information present in the report text.
- If a detail is not in the report, say it is not provided.
- Do not invent new numbers, claims, or company facts.
- Do not mention JSON, snapshots, code, or internal system implementation.

FORMAT:
- Plain text only.
- {roles}
- Use short paragraphs and spoken phrasing.
- Speak directly to the user using second-person language ("you", "your report").
- Avoid reading headings verbatim; translate them into spoken transitions.
- Do not output Markdown, bullet lists, tables, or headings.
- Aim for about {minutes} minutes of audio.

CONTENT REQUIREMENTS:
- Start with an engaging intro (what the user will learn).
- Follow the report in order and explicitly cover EVERY section:
    Executive Summary → Company/Product → Market → Strategy → Performance/Risks → Ops/Optimization → Explainability → Final Takeaways.
- For each section, include a brief but clear summary (roughly 2–4 sentences). Do not skip a section.
- Call out the most important numbers/statistics from the report when available.
- If a module was not executed (as stated in the report), explicitly mention it and avoid conclusions that depend on it.
- End with a concise recap and next steps addressed to the user.
- {disclaimer}

REPORT TEXT:
{report_text}
"""
