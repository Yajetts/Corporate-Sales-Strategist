from __future__ import annotations

from pathlib import Path
from typing import Optional


def write_simple_pdf(text: str, out_path: Path, title: Optional[str] = None) -> None:
    """Minimal PDF export (no new styling system).

    Uses reportlab if installed; otherwise raises ImportError.
    """

    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen.canvas import Canvas

    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = Canvas(str(out_path), pagesize=LETTER)
    width, height = LETTER

    x = 50
    y = height - 50
    line_height = 14

    if title:
        canvas.setFont("Helvetica-Bold", 14)
        canvas.drawString(x, y, title)
        y -= (line_height * 2)

    canvas.setFont("Helvetica", 10)

    for raw_line in text.splitlines() or [""]:
        line = raw_line.replace("\t", "    ")
        # Basic wrapping
        while len(line) > 110:
            canvas.drawString(x, y, line[:110])
            line = line[110:]
            y -= line_height
            if y < 60:
                canvas.showPage()
                canvas.setFont("Helvetica", 10)
                y = height - 50
        canvas.drawString(x, y, line)
        y -= line_height
        if y < 60:
            canvas.showPage()
            canvas.setFont("Helvetica", 10)
            y = height - 50

    canvas.save()
