"""
agent/tools.py

Five read-only tool functions used by the LangGraph agent.
Each tool pulls live data from existing SentinelML internals — no new
infrastructure required.

Tools
─────
  get_drift_report()          → latest drift_reports/ JSON artifact
  get_quality_report()        → latest quality_reports/ JSON artifact
  get_model_info()            → GET /model/info
  get_metrics_snapshot()      → GET /metrics/summary
  get_prediction_logs(hours)  → GET /logs?hours=N

All functions return plain dicts (JSON-serialisable) so LangGraph can
pass them directly as tool results to the LLM.

Stack notes
───────────
  - urllib.request only (no requests library, consistent with project)
  - langchain_core.tools.tool decorator for LangGraph compatibility
  - Completely free / open-source; zero paid dependencies
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from langchain_core.tools import tool

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_API_BASE = os.getenv("AGENT_API_BASE", "http://localhost:8000")
_ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
_HTTP_TIMEOUT = int(os.getenv("AGENT_HTTP_TIMEOUT", "10"))


# ── HTTP helper ───────────────────────────────────────────────────────────────


def _get(path: str) -> dict[str, Any]:
    """
    GET {_API_BASE}{path} and return parsed JSON.
    Uses stdlib urllib — no requests library.
    """
    url = f"{_API_BASE}{path}"
    try:
        with urlopen(url, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except URLError as exc:
        log.error("HTTP GET %s failed: %s", url, exc)
        return {"error": str(exc), "url": url}
    except json.JSONDecodeError as exc:
        log.error("JSON decode failed for %s: %s", url, exc)
        return {"error": f"Invalid JSON response: {exc}", "url": url}


# ── Artifact helpers ──────────────────────────────────────────────────────────


def _latest_json(subdir: str) -> dict[str, Any]:
    """
    Reads the most recently modified *.json file inside
    artifacts/{subdir}/ and returns its parsed contents.
    Returns an error dict if the directory or files don't exist.
    """
    report_dir = _ARTIFACTS_DIR / subdir
    if not report_dir.exists():
        return {"error": f"Directory not found: {report_dir}"}

    candidates = sorted(report_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return {"error": f"No JSON reports found in {report_dir}"}

    latest = candidates[-1]
    try:
        with open(latest) as f:
            data = json.load(f)
        data["_source_file"] = latest.name
        return data
    except Exception as exc:
        return {"error": f"Failed to read {latest}: {exc}"}


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool
def get_drift_report() -> dict[str, Any]:
    """
    Return the latest drift detection report.

    Reads the most recent JSON artifact from artifacts/drift_reports/.
    Includes overall drift flag, per-feature PSI scores, drifted feature
    list, severity breakdown, and prediction distribution drift.

    Use this when the user asks about:
      - whether drift is detected
      - which features are drifting
      - PSI scores or drift severity
      - whether retraining is needed due to data drift
    """
    report = _latest_json("drift_reports")
    if "error" in report:
        return report

    # Summarise for the LLM — keep it dense but readable
    drifted = report.get("drifted_features", [])
    summary = report.get("summary", {})
    prediction_drift = report.get("prediction_drift", {})

    return {
        "overall_drifted": report.get("overall_drifted", False),
        "model_version": report.get("model_version", "unknown"),
        "generated_at": report.get("generated_at"),
        "window_start": report.get("window_start"),
        "window_end": report.get("window_end"),
        "drifted_features": drifted,
        "drifted_feature_count": len(drifted),
        "total_features_checked": summary.get("total_features", 0),
        "drift_rate_pct": summary.get("drift_rate_pct", 0.0),
        "critical_count": summary.get("critical_count", 0),
        "warning_count": summary.get("warning_count", 0),
        "prediction_drift_detected": prediction_drift.get("drifted", False),
        "prediction_psi": prediction_drift.get("psi"),
        # Top-5 features by PSI for quick triage
        "top_drifted_features": sorted(
            [
                {
                    "feature": fr.get("feature"),
                    "psi": fr.get("psi"),
                    "severity": fr.get("severity"),
                    "method": fr.get("method"),
                }
                for fr in report.get("feature_results", [])
                if fr.get("drifted")
            ],
            key=lambda x: (x.get("psi") or 0),
            reverse=True,
        )[:5],
        "_source_file": report.get("_source_file"),
    }


@tool
def get_quality_report() -> dict[str, Any]:
    """
    Return the latest data quality report on the live prediction window.

    Reads the most recent JSON artifact from artifacts/quality_reports/.
    Includes missing value rates, out-of-range rates, unknown category rates,
    hard failures, and soft warnings per feature.

    Use this when the user asks about:
      - data quality of incoming predictions
      - missing values or anomalous inputs
      - whether quality checks are passing
      - feature-level issues in the prediction stream
    """
    report = _latest_json("quality_reports")
    if "error" in report:
        return report

    hard_failures = report.get("hard_failures", [])
    soft_warnings = report.get("soft_warnings", [])

    # Per-feature summary — top issues only
    feature_issues = [
        {
            "feature": fq.get("feature"),
            "missing_pct": fq.get("missing_pct"),
            "oor_pct": fq.get("oor_pct"),
            "unknown_cat_pct": fq.get("unknown_cat_pct"),
        }
        for fq in report.get("feature_quality", [])
        if (fq.get("missing_pct") or 0) > 0
        or (fq.get("oor_pct") or 0) > 0
        or (fq.get("unknown_cat_pct") or 0) > 0
    ]

    return {
        "overall_passed": report.get("overall_passed", False),
        "model_version": report.get("model_version", "unknown"),
        "generated_at": report.get("generated_at"),
        "window_start": report.get("window_start"),
        "window_end": report.get("window_end"),
        "window_size": report.get("window_size", 0),
        "hard_failure_count": len(hard_failures),
        "hard_failures": hard_failures,
        "soft_warning_count": len(soft_warnings),
        "soft_warnings": soft_warnings,
        "feature_issues": feature_issues,
        "_source_file": report.get("_source_file"),
    }


@tool
def get_model_info() -> dict[str, Any]:
    """Return current production model metadata including version, accuracy, and registration time."""
    registry_path = _ARTIFACTS_DIR / "production_model.json"
    if not registry_path.exists():
        return {"error": "production_model.json not found"}
    try:
        with open(registry_path) as f:
            return json.load(f)
    except Exception as exc:
        return {"error": str(exc)}


@tool
def get_metrics_snapshot() -> dict[str, Any]:
    """Return real-time snapshot of API prediction metrics including request counts, error rate, and latency."""
    try:
        from api.metrics import get_snapshot
        snapshot = get_snapshot()
        return snapshot.__dict__
    except Exception as exc:
        return {"error": str(exc)}


@tool
def get_prediction_logs(hours: int = 6) -> dict[str, Any]:
    """
    Return a summary of recent prediction logs.

    Calls GET /logs?hours={hours} on the live FastAPI service.
    Returns row count, class distribution, confidence stats, and a sample
    of recent records (up to 10) for the agent to cite.

    Args:
        hours: Look-back window in hours (default 6, max 168).

    Use this when the user asks about:
      - recent prediction activity
      - prediction volume in a time window
      - specific recent predictions or patterns
      - ground-truth labeling rate
    """
    hours = max(1, min(int(hours), 168))  # clamp 1–168
    try:
        from api.logger import query_logs
        from datetime import datetime, timedelta
        end = datetime.utcnow()
        start = end - timedelta(hours=hours)
        df = query_logs(start, end, limit=500)
        records = df.to_dict("records") if not df.empty else []
    except Exception as exc:
        return {"error": str(exc)}


    if not records:
        return {
            "hours": hours,
            "total_predictions": 0,
            "message": "No prediction logs found in this window.",
        }

    import statistics

    predictions = [r.get("prediction") for r in records if r.get("prediction") is not None]
    confidences = [r.get("confidence") for r in records if r.get("confidence") is not None]
    labeled = [r for r in records if r.get("ground_truth") is not None]

    class_counts = {0: predictions.count(0), 1: predictions.count(1)}

    return {
        "hours": hours,
        "total_predictions": len(records),
        "class_distribution": class_counts,
        "class_1_rate_pct": round(
            class_counts[1] / len(predictions) * 100, 2
        ) if predictions else 0,
        "mean_confidence": round(statistics.mean(confidences), 4) if confidences else None,
        "min_confidence": round(min(confidences), 4) if confidences else None,
        "max_confidence": round(max(confidences), 4) if confidences else None,
        "labeled_count": len(labeled),
        "label_rate_pct": round(len(labeled) / len(records) * 100, 2),
        # Sample of 10 most recent for the LLM to cite
        "recent_sample": [
            {
                "request_id": r.get("request_id"),
                "timestamp": r.get("timestamp"),
                "prediction": r.get("prediction"),
                "confidence": r.get("confidence"),
                "model_version": r.get("model_version"),
            }
            for r in records[:10]
        ],
    }


# ── Tool registry (imported by graph.py) ─────────────────────────────────────

TOOLS = [
    get_drift_report,
    get_quality_report,
    get_model_info,
    get_metrics_snapshot,
    get_prediction_logs,
]