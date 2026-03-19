from __future__ import annotations

from datetime import datetime, timezone
import secrets


def new_run_id(prefix: str = "pa") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = secrets.token_hex(4)
    return f"{prefix}_{ts}_{rand}"
