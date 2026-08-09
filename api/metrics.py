"""
api/metrics.py
==============
In-process metrics registry for the prediction API.

Responsibilities
----------------
- Maintain rolling counters and histograms for:
    * request throughput  (total, per model version)
    * prediction class distribution  (class_0 / class_1 counts)
    * confidence distribution        (histogram buckets)
    * latency distribution           (histogram buckets)
    * error counts                   (validation errors, inference errors)
    * ground-truth accuracy          (when labels arrive via label_outcome)
- Expose a metrics snapshot consumed by:
    * api/main.py                        → GET /metrics endpoint
    * monitoring/prometheus_exporter.py  → scraped by Prometheus
- Thread-safe — all mutation goes through a single RLock.

Public API
----------
    record_prediction(predicted_class, confidence, latency_ms, model_version) -> None
    record_prediction_from_result(result)  -> None
    record_from_log(log)                   -> None
    record_error(kind: str)                -> None
    record_label(prediction, ground_truth) -> None
    get_snapshot()                         -> MetricsSnapshot
    reset()                                -> None
    to_prometheus_text(snapshot=None)      -> str
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Histogram bucket boundaries
# ---------------------------------------------------------------------------
LATENCY_BUCKETS_MS: list[float] = [1, 5, 10, 25, 50, 100, 250, 500, 1000, float("inf")]
CONFIDENCE_BUCKETS: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, float("inf")]


# ---------------------------------------------------------------------------
# Histogram helper
# ---------------------------------------------------------------------------


class _Histogram:
    """
    Simple thread-unsafe histogram (caller holds the lock).
    Stores counts per upper-bound bucket plus sum and count for mean/percentile estimates.
    """

    def __init__(self, buckets: list[float]) -> None:
        self.buckets: list[float] = sorted(buckets)
        self.counts: list[int] = [0] * len(self.buckets)
        self.total_sum: float = 0.0
        self.total_count: int = 0

    def observe(self, value: float) -> None:
        self.total_sum += value
        self.total_count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def mean(self) -> float:
        return self.total_sum / self.total_count if self.total_count else 0.0

    def percentile(self, p: float) -> float:
        """Linear-interpolation percentile estimate. p in [0, 100]."""
        if self.total_count == 0:
            return 0.0
        target = math.ceil(p / 100.0 * self.total_count)
        cumulative = 0
        prev_bound = 0.0
        for bound, count in zip(self.buckets, self.counts):
            cumulative += count
            if cumulative >= target:
                return bound if not math.isinf(bound) else prev_bound
            prev_bound = bound if not math.isinf(bound) else prev_bound
        return prev_bound

    def to_dict(self) -> dict:
        return {
            "buckets": [
                {"le": b if not math.isinf(b) else "+Inf", "count": c}
                for b, c in zip(self.buckets, self.counts)
            ],
            "sum": self.total_sum,
            "count": self.total_count,
            "mean": self.mean(),
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
        }

    def reset(self) -> None:
        self.counts = [0] * len(self.buckets)
        self.total_sum = 0.0
        self.total_count = 0


# ---------------------------------------------------------------------------
# MetricsSnapshot — immutable view returned to callers
# ---------------------------------------------------------------------------


@dataclass
class MetricsSnapshot:
    """
    Point-in-time read of all metrics.
    Consumed by main.py (/metrics endpoint) and prometheus_exporter.py.
    """

    requests_total: int
    errors_total: int
    predictions_class_0: int
    predictions_class_1: int

    requests_by_version: dict[str, int]
    errors_by_kind: dict[str, int]

    latency_histogram: dict
    confidence_histogram: dict

    labeled_total: int
    correct_total: int
    accuracy: float  # 0.0 when labeled_total == 0

    window_start: float  # Unix timestamp of first recorded event
    window_end: float  # Unix timestamp of snapshot

    request_rate: float  # requests/sec over window
    error_rate: float  # errors/sec over window

    def to_dict(self) -> dict:
        return {
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "predictions_class_0": self.predictions_class_0,
            "predictions_class_1": self.predictions_class_1,
            "requests_by_version": self.requests_by_version,
            "errors_by_kind": self.errors_by_kind,
            "latency_ms": self.latency_histogram,
            "confidence": self.confidence_histogram,
            "labeled_total": self.labeled_total,
            "correct_total": self.correct_total,
            "accuracy": self.accuracy,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "request_rate_per_sec": round(self.request_rate, 4),
            "error_rate_per_sec": round(self.error_rate, 4),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class _MetricsRegistry:
    """Thread-safe, in-process metrics store."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reset_state()

    def _reset_state(self) -> None:
        self._requests_total: int = 0
        self._errors_total: int = 0
        self._predictions_class_0: int = 0
        self._predictions_class_1: int = 0
        self._requests_by_version: dict[str, int] = defaultdict(int)
        self._errors_by_kind: dict[str, int] = defaultdict(int)
        self._latency_hist = _Histogram(LATENCY_BUCKETS_MS)
        self._confidence_hist = _Histogram(CONFIDENCE_BUCKETS)
        self._labeled_total: int = 0
        self._correct_total: int = 0
        self._window_start: float = time.time()

    def record_prediction(
        self,
        prediction: int,
        confidence: float,
        latency_ms: float,
        model_version: str,
    ) -> None:
        """
        Record a successful inference event.
        For errors, call record_error() instead — do NOT use this with error=True
        to avoid double-counting requests_total.
        """
        with self._lock:
            self._requests_total += 1
            self._requests_by_version[model_version] += 1
            self._latency_hist.observe(latency_ms)

            if prediction == 0:
                self._predictions_class_0 += 1
            else:
                self._predictions_class_1 += 1

            self._confidence_hist.observe(confidence)

    def record_from_log(self, log: Any) -> None:
        """
        Replay a PredictionLog into the registry.
        Used by prometheus_exporter.py to bootstrap metrics from stored logs.
        """
        try:
            if log.prediction == -1:
                # Sentinel value from predict_batch error path
                self.record_error("inference_error")
                return
            self.record_prediction(
                prediction=log.prediction,
                confidence=log.confidence,
                latency_ms=log.latency_ms,
                model_version=log.model_version,
            )
            if log.ground_truth is not None:
                self.record_label(
                    prediction=log.prediction,
                    ground_truth=log.ground_truth,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("record_from_log failed: %s", exc)

    def record_error(self, kind: str = "unknown") -> None:
        """
        Record a named error (e.g. "validation_error", "inference_error").
        Increments requests_total once — do NOT also call record_prediction()
        for the same request.
        """
        with self._lock:
            self._requests_total += 1
            self._errors_total += 1
            self._errors_by_kind[kind] += 1

    def record_label(self, prediction: int, ground_truth: int) -> None:
        """Update accuracy counters when a ground-truth label arrives."""
        with self._lock:
            self._labeled_total += 1
            if prediction == ground_truth:
                self._correct_total += 1

    def get_snapshot(self) -> MetricsSnapshot:
        with self._lock:
            now = time.time()
            elapsed = max(now - self._window_start, 1e-9)
            accuracy = (
                self._correct_total / self._labeled_total
                if self._labeled_total > 0
                else 0.0
            )
            return MetricsSnapshot(
                requests_total=self._requests_total,
                errors_total=self._errors_total,
                predictions_class_0=self._predictions_class_0,
                predictions_class_1=self._predictions_class_1,
                requests_by_version=dict(self._requests_by_version),
                errors_by_kind=dict(self._errors_by_kind),
                latency_histogram=self._latency_hist.to_dict(),
                confidence_histogram=self._confidence_hist.to_dict(),
                labeled_total=self._labeled_total,
                correct_total=self._correct_total,
                accuracy=accuracy,
                window_start=self._window_start,
                window_end=now,
                request_rate=self._requests_total / elapsed,
                error_rate=self._errors_total / elapsed,
            )

    def reset(self) -> None:
        with self._lock:
            self._reset_state()
        logger.info("MetricsRegistry reset")


# ---------------------------------------------------------------------------
# Module-level singleton + public functions
# ---------------------------------------------------------------------------

_registry = _MetricsRegistry()


def record_prediction(
    predicted_class: int,
    confidence: float,
    latency_ms: float,
    model_version: str,
) -> None:
    """Record a successful inference outcome into the global registry."""
    _registry.record_prediction(
        prediction=predicted_class,
        confidence=confidence,
        latency_ms=latency_ms,
        model_version=model_version,
    )


def record_prediction_from_result(result: Any) -> None:
    """
    Convenience wrapper — accepts a PredictionResult from predict.py directly.
    Used by main.py on every successful inference call.
    """
    _registry.record_prediction(
        prediction=result.prediction,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        model_version=result.model_version,
    )


def record_from_log(log: Any) -> None:
    """Replay a PredictionLog into the global registry."""
    _registry.record_from_log(log)


def record_error(kind: str = "unknown") -> None:
    """Record a named error event. Do NOT also call record_prediction() for the same request."""
    _registry.record_error(kind)


def record_label(prediction: int, ground_truth: int) -> None:
    """Update accuracy counters when ground truth is known."""
    _registry.record_label(prediction, ground_truth)


def get_snapshot() -> MetricsSnapshot:
    """Return the current MetricsSnapshot."""
    return _registry.get_snapshot()


def reset() -> None:
    """Zero all counters (tests / admin)."""
    _registry.reset()


# ---------------------------------------------------------------------------
# Prometheus text-format export
# ---------------------------------------------------------------------------


def to_prometheus_text(snapshot: MetricsSnapshot | None = None) -> str:
    """
    Render the current metrics snapshot in Prometheus text exposition format.
    Metric names use prefix ml_api_ — matches prometheus.yml scrape config.
    Called by prometheus_exporter.py and served directly at GET /metrics.
    """
    s = snapshot or get_snapshot()
    lines: list[str] = []

    def _gauge(name: str, value: float, labels: str = "") -> None:
        label_str = f"{{{labels}}}" if labels else ""
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{label_str} {value}")

    def _counter(name: str, value: float, labels: str = "") -> None:
        label_str = f"{{{labels}}}" if labels else ""
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name}{label_str} {value}")

    # Totals
    _counter("ml_api_requests_total", s.requests_total)
    _counter("ml_api_errors_total", s.errors_total)
    _counter("ml_api_predictions_class_0_total", s.predictions_class_0)
    _counter("ml_api_predictions_class_1_total", s.predictions_class_1)

    # Per-version
    for version, count in s.requests_by_version.items():
        _counter(
            "ml_api_requests_by_version_total", count, f'model_version="{version}"'
        )

    # Per-error-kind
    for kind, count in s.errors_by_kind.items():
        _counter("ml_api_errors_by_kind_total", count, f'kind="{kind}"')

    # Latency histogram
    lines.append("# TYPE ml_api_latency_ms histogram")
    for bucket in s.latency_histogram["buckets"]:
        le = bucket["le"]
        lines.append(f'ml_api_latency_ms_bucket{{le="{le}"}} {bucket["count"]}')
    lines.append(f'ml_api_latency_ms_sum {s.latency_histogram["sum"]}')
    lines.append(f'ml_api_latency_ms_count {s.latency_histogram["count"]}')

    # Confidence histogram
    lines.append("# TYPE ml_api_confidence histogram")
    for bucket in s.confidence_histogram["buckets"]:
        le = bucket["le"]
        lines.append(f'ml_api_confidence_bucket{{le="{le}"}} {bucket["count"]}')
    lines.append(f'ml_api_confidence_sum {s.confidence_histogram["sum"]}')
    lines.append(f'ml_api_confidence_count {s.confidence_histogram["count"]}')

    # Accuracy
    _gauge("ml_api_accuracy", s.accuracy)
    _gauge("ml_api_labeled_total", s.labeled_total)

    # Rates
    _gauge("ml_api_request_rate_per_sec", s.request_rate)
    _gauge("ml_api_error_rate_per_sec", s.error_rate)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Module self-test (python -m api.metrics)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    reset()

    for i in range(5):
        record_prediction(
            predicted_class=i % 2,
            confidence=0.6 + i * 0.05,
            latency_ms=10 + i * 3,
            model_version="v_test",
        )

    record_error("validation_error")
    record_error("inference_error")

    record_label(1, 1)
    record_label(0, 0)
    record_label(1, 0)

    snap = get_snapshot()
    assert snap.requests_total == 7, snap.requests_total
    assert snap.errors_total == 2, snap.errors_total
    assert snap.predictions_class_0 == 3
    assert snap.predictions_class_1 == 2
    assert snap.labeled_total == 3
    assert snap.correct_total == 2
    assert abs(snap.accuracy - 2 / 3) < 1e-9
    assert snap.latency_histogram["count"] == 5
    assert snap.requests_by_version["v_test"] == 7
    assert snap.errors_by_kind["validation_error"] == 1

    prom = to_prometheus_text(snap)
    assert "ml_api_requests_total 7" in prom
    assert "ml_api_errors_total 2" in prom
    assert "ml_api_latency_ms_bucket" in prom
    assert "ml_api_accuracy" in prom

    reset()
    assert get_snapshot().requests_total == 0

    logger.info("ALL SELF-TESTS PASSED")
    sys.exit(0)
