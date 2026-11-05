"""
experiments/mlflow_utils.py
---------------------------
Fail-open MLflow tracking helper for SentinelML.

Design rules
------------
1. Tracking is OBSERVATIONAL. It must never break training, evaluation,
   an Airflow DAG, or an experiment script. Every public function swallows
   all exceptions and returns a harmless value (False / None).
2. Fast failure. A one-time stdlib preflight (GET <uri>/health, 2s timeout)
   decides whether tracking is enabled for this process. MLflow's own HTTP
   retries are capped so a dead server can never stall a pipeline.
3. Optional dependency. If `mlflow` is not installed in the image, every
   call is a no-op.
4. The JSON registry (artifacts/production_model.json) stays authoritative
   for serving. MLflow runs only RECORD what happened and are linked to the
   registry by the `version_tag` tag.

Public API
----------
    setup_tracking(experiment_name) -> bool
    log_run(params=None, metrics=None, artifacts=None, tags=None,
            run_name=None, experiment_name=None) -> Optional[str]   # run_id

Standard experiment names (one per task):
    sentinelml-training              train.py + evaluate.py runs
    sentinelml-task1-drift-recovery
    sentinelml-task2-detector-eval
    sentinelml-task3-model-eval
    sentinelml-task4-load-test
    sentinelml-task5-agent-eval
"""

from __future__ import annotations

import logging
import math
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# Cap MLflow's internal HTTP retries BEFORE mlflow is imported anywhere.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

logger = logging.getLogger("experiments.mlflow_utils")

DEFAULT_URI = (
    "http://localhost:5000"  # host default; containers set MLFLOW_TRACKING_URI
)
PREFLIGHT_TIMEOUT_S = 2.0
_PARAM_MAX_LEN = 500
_KEY_BAD_CHARS = re.compile(r"[^0-9A-Za-z_\-\. /]")

# None = not checked yet, True/False = cached preflight result for this process
_enabled: Optional[bool] = None
_current_experiment: Optional[str] = None


# ----------------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------------
def _tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI).rstrip("/")


def _server_reachable(uri: str) -> bool:
    try:
        with urllib.request.urlopen(f"{uri}/health", timeout=PREFLIGHT_TIMEOUT_S) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _clean_key(key: Any) -> str:
    return _KEY_BAD_CHARS.sub("_", str(key))[:250]


def _clean_metrics(metrics: Optional[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (metrics or {}).items():
        try:
            if isinstance(v, bool):
                v = int(v)
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[_clean_key(k)] = f
    return out


def _clean_params(params: Optional[Mapping[str, Any]]) -> dict[str, str]:
    return {
        _clean_key(k): str(v)[:_PARAM_MAX_LEN]
        for k, v in (params or {}).items()
        if v is not None
    }


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def setup_tracking(experiment_name: str) -> bool:
    """
    Point MLflow at the tracking server and select (or create) an experiment.
    Returns True if tracking is live, False if disabled. Never raises.
    """
    global _enabled, _current_experiment
    try:
        if _enabled is None:
            uri = _tracking_uri()
            _enabled = _server_reachable(uri)
            if not _enabled:
                logger.warning(
                    "MLflow unreachable at %s - tracking disabled for this process.",
                    uri,
                )
        if not _enabled:
            return False

        import mlflow  # lazy: missing package => no-op

        mlflow.set_tracking_uri(_tracking_uri())
        mlflow.set_experiment(experiment_name)
        _current_experiment = experiment_name
        return True
    except Exception as exc:  # noqa: BLE001
        _enabled = False
        logger.warning("MLflow setup failed (%s) - tracking disabled.", exc)
        return False


def log_run(
    params: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Iterable[str | Path]] = None,
    tags: Optional[Mapping[str, Any]] = None,
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Optional[str]:
    """
    Log one complete run (params, metrics, artifacts, tags) in a single call.
    Returns the MLflow run_id, or None if tracking is disabled / failed.
    Never raises.

    Always pass tags={"version_tag": ...} for training/eval runs so each MLflow
    run maps 1:1 to its JSON-registry entry.
    """
    try:
        if experiment_name is not None or _current_experiment is None:
            if not setup_tracking(experiment_name or "sentinelml-default"):
                return None
        elif not _enabled:
            return None

        import mlflow

        all_tags = {_clean_key(k): str(v)[:5000] for k, v in (tags or {}).items()}
        all_tags.setdefault("sentinelml.source", "experiments.mlflow_utils")

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(all_tags)
            clean_params = _clean_params(params)
            if clean_params:
                mlflow.log_params(clean_params)
            clean_metrics = _clean_metrics(metrics)
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)
            for art in artifacts or []:
                p = Path(art)
                try:
                    if p.is_dir():
                        mlflow.log_artifacts(str(p), artifact_path=p.name)
                    elif p.is_file():
                        mlflow.log_artifact(str(p))
                    else:
                        logger.warning("MLflow artifact not found, skipped: %s", p)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("MLflow artifact upload failed for %s: %s", p, exc)
            return run.info.run_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow log_run failed (%s) - continuing without tracking.", exc)
        return None
