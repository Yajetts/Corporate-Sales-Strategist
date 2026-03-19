"""Post-analysis report & audio summary layer.

This package is intentionally loosely coupled: it only *reads* existing module outputs
and produces post-run artifacts (snapshot, report, audio) for the dashboard.
"""

from .snapshot import AnalysisSnapshot
