"""One-command local runner that starts the API without any DB dependencies.

Usage:
  python -m src.api.run_nodb --host 0.0.0.0 --port 5000

This sets DB_INIT_DISABLE=1 before importing the Flask app factory so the
process doesn't attempt to connect to Postgres/Mongo.
"""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sales Strategist API (no DB mode)")
    parser.add_argument("--host", default=os.getenv("API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "5000")))
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not automatically open the dashboard URL in a browser",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode (default: off)",
    )
    args = parser.parse_args()

    # Disable DB init/health checks before importing the app module.
    os.environ.setdefault("DB_INIT_DISABLE", "1")

    # Also keep post-analysis synchronous by default for local runs.
    os.environ.setdefault("POST_ANALYSIS_USE_CELERY", "0")

    # Importing `src.api.app` creates the Flask app instance (`app`) at import time.
    from src.api.app import app

    open_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    dashboard_url = f"http://{open_host}:{args.port}/dashboard/"

    if not args.no_open_browser:
        # Delay slightly so the dev server has time to bind before browser open.
        threading.Timer(0.8, lambda: webbrowser.open(dashboard_url)).start()

    print(f"API root: http://{open_host}:{args.port}/")
    print(f"Dashboard: {dashboard_url}")

    app.run(host=args.host, port=args.port, debug=bool(args.debug))


if __name__ == "__main__":
    main()
