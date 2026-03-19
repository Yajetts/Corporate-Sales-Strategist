"""REST API package.

Intentionally avoids importing the Flask app at import time to prevent circular
imports (e.g., modules that only need database utilities importing
`src.api.database`).
"""

__all__ = []
