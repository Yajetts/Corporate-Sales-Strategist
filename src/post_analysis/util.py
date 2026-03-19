from __future__ import annotations

import json
from typing import Any, Dict


def safe_json_dumps(obj: Any, max_chars: int = 18000) -> str:
    """Dump JSON with a hard size cap to avoid blowing up prompts.

    If it exceeds max_chars, it is truncated with a clear note.
    """

    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    head = text[: max_chars]
    return head + "\n\n...TRUNCATED: snapshot too large for prompt...\n"


def coalesce(*values):
    for v in values:
        if v is not None:
            return v
    return None
