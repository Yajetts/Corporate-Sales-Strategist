from __future__ import annotations

from typing import Optional


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _minimal_markdown_to_html(md: str) -> str:
    """Very small Markdown renderer for the dashboard.

    This is intentionally minimal and designed to render our strict report template
    consistently when the optional `markdown` package isn't installed.
    """

    lines = md.splitlines()
    out: list[str] = []

    in_ul = False
    in_ol = False

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    for raw in lines:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if not stripped:
            close_lists()
            continue

        if stripped.startswith("# "):
            close_lists()
            out.append(f"<h1>{_escape_html(stripped[2:].strip())}</h1>")
            continue

        if stripped.startswith("## "):
            close_lists()
            out.append(f"<h2>{_escape_html(stripped[3:].strip())}</h2>")
            continue

        if stripped.startswith("- "):
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_escape_html(stripped[2:].strip())}</li>")
            continue

        # Basic ordered list: "1. ..."
        if len(stripped) > 3 and stripped[0].isdigit() and stripped[1:3] == ". ":
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{_escape_html(stripped[3:].strip())}</li>")
            continue

        close_lists()
        out.append(f"<p>{_escape_html(stripped)}</p>")

    close_lists()
    return "\n".join(out)


def markdown_to_html(md: str) -> str:
    """Convert Markdown to HTML for dashboard rendering."""

    try:
        import markdown
    except ImportError as e:
        # Fall back to a tiny renderer so the dashboard stays consistent.
        return _minimal_markdown_to_html(md)

    return markdown.markdown(
        md,
        extensions=["tables", "fenced_code", "toc"],
        output_format="html5",
    )


def guess_content_type(path: str) -> Optional[str]:
    lower = path.lower()
    if lower.endswith(".md"):
        return "text/markdown"
    if lower.endswith(".txt"):
        return "text/plain"
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".mp3"):
        return "audio/mpeg"
    return None
