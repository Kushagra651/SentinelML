"""
monitoring/prometheus_exporter.py

Serves Prometheus text-format metrics on port 9100 by aggregating three sources:

  1. api/metrics.py        → in-process prediction counters / latency histograms
  2. drift_report.py       → latest saved drift report (PSI, drift flags, counts)
  3. quality_report.py     → latest saved quality report (missing rates, check pass/fail)

Metric namespaces:
  ml_api_*     — from api/metrics.py (prediction-time counters, latency)
  ml_monitor_* — from drift + quality reports (this exporter)

Run modes:
  Standalone HTTP server (default):  python -m monitoring.prometheus_exporter
  Single-shot stdout dump:           python -m monitoring.prometheus_exporter --once

Prometheus scrape target: ml_api:9100/metrics  (see prometheus/prometheus.yml)
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import List, Optional

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "9100"))
SCRAPE_INTERVAL = int(os.getenv("SCRAPE_INTERVAL_SEC", "30"))
NAMESPACE = os.getenv("METRICS_NAMESPACE", "ml_monitor")


# ── Prometheus text helpers ───────────────────────────────────────────────────


def _label_str(labels: Optional[dict]) -> str:
    if not labels:
        return ""
    pairs = ",".join(f'{k}="{v}"' for k, v in labels.items())
    return f"{{{pairs}}}"


def _gauge(
    name: str, value: float, labels: Optional[dict] = None, help_text: str = ""
) -> str:
    """Emit HELP + TYPE + metric line for a gauge."""
    full = f"{NAMESPACE}_{name}"
    lines = []
    if help_text:
        lines.append(f"# HELP {full} {help_text}")
    lines.append(f"# TYPE {full} gauge")
    lines.append(f"{full}{_label_str(labels)} {value}")
    return "\n".join(lines)


def _report_age_seconds(generated_at: str) -> Optional[float]:
    """
    Returns seconds since generated_at. Handles both naive and aware ISO strings.
    Returns None on any parse failure — callers skip the metric rather than crash.
    """
    try:
        gen = datetime.fromisoformat(generated_at)
        if gen.tzinfo is None:
            # Treat naive timestamps as UTC (all our report writers use utcnow())
            gen = gen.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - gen).total_seconds()
    except Exception:
        return None


# ── Source 1: API in-process metrics ─────────────────────────────────────────


def _collect_api_metrics() -> str:
    """
    Pulls from api/metrics.py's in-memory registry via to_prometheus_text().
    This import is intentionally lazy — the exporter runs in the same process
    as the API (started by Dockerfile.api CMD), so the import always succeeds
    at runtime. Falls back to empty string if somehow unavailable.
    """
    try:
        from api.metrics import to_prometheus_text  # noqa: PLC0415

        return to_prometheus_text()
    except Exception as e:
        log.debug("API metrics unavailable: %s", e)
        return ""


# ── Source 2: Drift report metrics ───────────────────────────────────────────


def _collect_drift_metrics() -> str:
    try:
        from monitoring.drift_report import load_latest_drift_report  # noqa: PLC0415

        report = load_latest_drift_report()
    except Exception as e:
        log.error("Failed to load drift report: %s", e)
        return ""

    if report is None:
        log.debug("No drift report on disk yet.")
        return ""

    lines: List[str] = []
    mv = {"model_version": report.model_version}

    lines.append(
        _gauge(
            "drift_overall",
            float(report.overall_drifted),
            labels=mv,
            help_text="1 if overall drift detected in the latest window, else 0.",
        )
    )
    lines.append(
        _gauge(
            "drift_feature_count",
            float(len(report.drifted_features)),
            labels=mv,
            help_text="Number of features with detected drift.",
        )
    )
    lines.append(
        _gauge(
            "drift_rate_pct",
            float(report.summary.get("drift_rate_pct", 0.0)),
            labels=mv,
            help_text="Percentage of checked features that drifted.",
        )
    )
    lines.append(
        _gauge(
            "drift_critical_count",
            float(report.summary.get("critical_count", 0)),
            help_text="Features with critical drift severity.",
        )
    )
    lines.append(
        _gauge(
            "drift_warning_count",
            float(report.summary.get("warning_count", 0)),
            help_text="Features with warning drift severity.",
        )
    )

    # Per-feature PSI (one series per feature)
    full_psi = f"{NAMESPACE}_drift_feature_psi"
    lines.append(f"# HELP {full_psi} PSI score per feature in the latest drift window.")
    lines.append(f"# TYPE {full_psi} gauge")
    for fr in report.feature_results:
        if fr.psi is not None:
            lines.append(
                f'{full_psi}{{feature="{fr.feature}",severity="{fr.severity}"}} {fr.psi}'
            )

    # Per-feature drift flag
    full_flag = f"{NAMESPACE}_drift_feature_drifted"
    lines.append(f"# HELP {full_flag} 1 if feature drifted in the latest window.")
    lines.append(f"# TYPE {full_flag} gauge")
    for fr in report.feature_results:
        lines.append(
            f'{full_flag}{{feature="{fr.feature}",method="{fr.method}"}} {float(fr.drifted)}'
        )

    # Prediction distribution drift
    pd_ = report.prediction_drift
    lines.append(
        _gauge(
            "drift_prediction",
            float(pd_.drifted),
            labels=mv,
            help_text="1 if prediction distribution drift detected.",
        )
    )
    if pd_.psi is not None:
        lines.append(
            _gauge(
                "drift_prediction_psi",
                pd_.psi,
                help_text="PSI of prediction class distribution.",
            )
        )

    # Report staleness
    age = _report_age_seconds(report.generated_at)
    if age is not None:
        lines.append(
            _gauge(
                "drift_report_age_seconds",
                age,
                help_text="Seconds since the latest drift report was generated.",
            )
        )

    return "\n".join(lines)


# ── Source 3: Quality report metrics ─────────────────────────────────────────


def _collect_quality_metrics() -> str:
    try:
        from monitoring.quality_report import (
            load_latest_quality_report,
        )  # noqa: PLC0415

        report = load_latest_quality_report()
    except Exception as e:
        log.error("Failed to load quality report: %s", e)
        return ""

    if report is None:
        log.debug("No quality report on disk yet.")
        return ""

    lines: List[str] = []
    mv = {"model_version": report.model_version}

    lines.append(
        _gauge(
            "quality_overall_passed",
            float(report.overall_passed),
            labels=mv,
            help_text="1 if the latest quality report passed all hard checks.",
        )
    )
    lines.append(
        _gauge(
            "quality_hard_failures",
            float(len(report.hard_failures)),
            help_text="Number of hard check failures in the latest quality report.",
        )
    )
    lines.append(
        _gauge(
            "quality_soft_warnings",
            float(len(report.soft_warnings)),
            help_text="Number of soft warnings in the latest quality report.",
        )
    )
    lines.append(
        _gauge(
            "quality_window_size",
            float(report.window_size),
            help_text="Number of prediction records in the quality report window.",
        )
    )

    # Per-feature missing rate
    full_miss = f"{NAMESPACE}_quality_feature_missing_pct"
    lines.append(
        f"# HELP {full_miss} Missing value rate per feature in quality window."
    )
    lines.append(f"# TYPE {full_miss} gauge")
    for fq in report.feature_quality:
        lines.append(f'{full_miss}{{feature="{fq.feature}"}} {fq.missing_pct}')

    # Per-feature out-of-range rate (numerical features only)
    full_oor = f"{NAMESPACE}_quality_feature_oor_pct"
    lines.append(f"# HELP {full_oor} Out-of-range rate for numerical features.")
    lines.append(f"# TYPE {full_oor} gauge")
    for fq in report.feature_quality:
        if fq.oor_pct is not None:
            lines.append(f'{full_oor}{{feature="{fq.feature}"}} {fq.oor_pct}')

    # Per-feature unknown category rate (categorical features only)
    full_unk = f"{NAMESPACE}_quality_feature_unknown_cat_pct"
    lines.append(f"# HELP {full_unk} Unknown category rate for categorical features.")
    lines.append(f"# TYPE {full_unk} gauge")
    for fq in report.feature_quality:
        if fq.unknown_cat_pct is not None:
            lines.append(f'{full_unk}{{feature="{fq.feature}"}} {fq.unknown_cat_pct}')

    # Per-check pass/fail
    full_chk = f"{NAMESPACE}_quality_check_passed"
    lines.append(f"# HELP {full_chk} 1 if the named quality check passed.")
    lines.append(f"# TYPE {full_chk} gauge")
    for chk in report.checks:
        lines.append(
            f'{full_chk}{{check="{chk.check_name}",severity="{chk.severity}"}} {float(chk.passed)}'
        )

    # Report staleness
    age = _report_age_seconds(report.generated_at)
    if age is not None:
        lines.append(
            _gauge(
                "quality_report_age_seconds",
                age,
                help_text="Seconds since the latest quality report was generated.",
            )
        )

    return "\n".join(lines)


# ── Aggregator ────────────────────────────────────────────────────────────────


def collect_all_metrics() -> str:
    """
    Collects from all three sources and returns a valid Prometheus text payload.
    Each collector is isolated — one failure never affects the others.
    """
    parts = []
    for label, fn in [
        ("api", _collect_api_metrics),
        ("drift", _collect_drift_metrics),
        ("quality", _collect_quality_metrics),
    ]:
        try:
            text = fn()
            if text:
                parts.append(text)
        except Exception as e:
            log.error("Collector '%s' crashed unexpectedly: %s", label, e)

    return "\n\n".join(parts) + "\n"


# ── HTTP server ───────────────────────────────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        try:
            payload = collect_all_metrics().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            log.error("Handler error: %s", e)
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, fmt, *args):  # suppress noisy access logs
        log.debug(fmt, *args)


class PrometheusExporter:
    """Thin wrapper around HTTPServer. Runs in a daemon thread."""

    def __init__(self, port: int = EXPORTER_PORT):
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        self._server = HTTPServer(("0.0.0.0", self.port), _Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        log.info("Prometheus exporter listening on :%d/metrics", self.port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
        log.info("Prometheus exporter stopped.")


# ── CLI entry point ───────────────────────────────────────────────────────────


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="ML Monitoring Prometheus Exporter")
    parser.add_argument(
        "--once", action="store_true", help="Print metrics once to stdout and exit."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=EXPORTER_PORT,
        help=f"HTTP port (default: {EXPORTER_PORT})",
    )
    args = parser.parse_args()

    if args.once:
        print(collect_all_metrics())
        return

    exporter = PrometheusExporter(port=args.port)
    exporter.start()
    try:
        while True:
            time.sleep(SCRAPE_INTERVAL)
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
        exporter.stop()


if __name__ == "__main__":
    _main()
