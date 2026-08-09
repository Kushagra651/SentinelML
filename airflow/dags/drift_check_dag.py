"""
airflow/dags/drift_check_dag.py

Periodic drift + quality check on live prediction window:
  load_reference → fetch_live_window → compute_drift → compute_quality → gate

Schedule : DRIFT_CHECK_SCHEDULE env (default: every 6 hours)

Gate task writes Airflow Variable `drift_retrain_triggered` (true/false).
retrain_trigger_dag reads this variable 30 min later.

Design notes:
  - model_registry.json["models"] is a dict keyed by version_tag, not a list.
  - logger.py has no get_logger(); query_logs() is a module-level function.
  - PredictionLog field is "prediction" (int), not "predicted_class".
  - drift_report and quality_report run in parallel after fetch_live_window.
  - push_metrics task is removed — prometheus_exporter reads report files on
    its own scrape cycle; calling collect_all_metrics() from a DAG task adds
    no value and couples Airflow to the exporter unnecessarily.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

SCHEDULE = os.getenv("DRIFT_CHECK_SCHEDULE", "0 */6 * * *")
WINDOW_HOURS = int(os.getenv("DRIFT_WINDOW_HOURS", "6"))
MIN_WINDOW_ROWS = int(os.getenv("DRIFT_MIN_SAMPLES", "30"))
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
RETRAIN_VAR_KEY = "drift_retrain_triggered"

# Metadata columns in prediction logs — excluded from feature drift comparison
_LOG_META_COLS = {
    "request_id",
    "timestamp",
    "model_version",
    "model_alias",
    "prediction",
    "probability_class_0",
    "probability_class_1",
    "confidence",
    "latency_ms",
    "ground_truth",
    "warnings",
    "schema_version",
}

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "execution_timeout": timedelta(minutes=30),
}


# ── Task callables ────────────────────────────────────────────────────────────


def task_load_reference(**ctx):
    """
    Reads model_registry.json to find the production version tag, then loads
    the corresponding reference features parquet (saved during training).
    Pushes ref_path and model_version to XCom.
    """
    import pandas as pd

    reg_path = Path(ARTIFACTS_DIR) / "models" / "model_registry.json"
    if not reg_path.exists():
        raise FileNotFoundError(f"Model registry not found: {reg_path}")

    with open(reg_path) as f:
        registry = json.load(f)

    # registry["models"] is a dict: {version_tag: entry_dict}
    prod_version = registry.get("production_version")
    if not prod_version:
        raise RuntimeError("No production_version set in model_registry.json.")

    # Reference dataset: feature store parquet saved by ingest.py during training
    ref_path = Path(ARTIFACTS_DIR) / "feature_store" / "train.parquet"
    if not ref_path.exists():
        # Fallback: most recent features parquet anywhere under artifacts
        candidates = sorted(Path(ARTIFACTS_DIR).rglob("*.parquet"), reverse=True)
        candidates = [p for p in candidates if "live_window" not in p.name]
        if not candidates:
            raise FileNotFoundError(
                "No reference features parquet found in artifacts/."
            )
        ref_path = candidates[0]
        log.warning("Exact reference not found, using fallback: %s", ref_path)

    ref_df = pd.read_parquet(ref_path)

    ti = ctx["ti"]
    ti.xcom_push(key="ref_path", value=str(ref_path))
    ti.xcom_push(key="model_version", value=prod_version)
    ti.xcom_push(key="ref_size", value=len(ref_df))
    log.info("Reference loaded: %d rows  model_version=%s", len(ref_df), prod_version)


def task_fetch_live_window(**ctx):
    """
    Pulls prediction logs for the last WINDOW_HOURS from PostgreSQL via
    logger.query_logs(). Falls back to JSONL files if DB is unreachable.
    Raises if fewer than MIN_WINDOW_ROWS rows — not enough signal for drift.
    """
    import pandas as pd

    ti = ctx["ti"]
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=WINDOW_HOURS)
    window_end = now

    df = pd.DataFrame()

    try:
        from api.logger import query_logs  # noqa: PLC0415

        df = query_logs(
            start=window_start,
            end=window_end,
            limit=50_000,
        )
        log.info("Fetched %d rows from PostgreSQL logger.", len(df))
    except Exception as e:
        log.warning("Logger DB unavailable (%s). Falling back to JSONL files.", e)
        rows = []
        jsonl_files = sorted(glob.glob(f"{ARTIFACTS_DIR}/logs/*.jsonl"), reverse=True)[
            :7
        ]
        for path in jsonl_files:
            with open(path) as fh:
                for line in fh:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if rows:
            df = pd.DataFrame(rows)
            if "timestamp" in df.columns:
                df = df[df["timestamp"] >= window_start.isoformat()]
        log.info("JSONL fallback: %d rows after time filter.", len(df))

    if len(df) < MIN_WINDOW_ROWS:
        raise ValueError(
            f"Live window has only {len(df)} rows (need ≥ {MIN_WINDOW_ROWS}). "
            "Not enough data for a reliable drift check — skipping."
        )

    live_path = f"{ARTIFACTS_DIR}/live_window_{ctx['ds_nodash']}.parquet"
    df.to_parquet(live_path, index=False)

    ti.xcom_push(key="live_path", value=live_path)
    ti.xcom_push(key="window_start", value=window_start.isoformat())
    ti.xcom_push(key="window_end", value=window_end.isoformat())
    ti.xcom_push(key="live_size", value=len(df))
    log.info(
        "Live window saved: %d rows [%s → %s] → %s",
        len(df),
        window_start.isoformat(),
        window_end.isoformat(),
        live_path,
    )


def task_compute_drift(**ctx):
    """
    Computes feature drift between the training reference set and the live window.
    Only passes feature columns (strips log metadata) to compute_drift_report().
    """
    import pandas as pd
    from monitoring.drift_report import compute_drift_report  # noqa: PLC0415

    ti = ctx["ti"]
    ref_df = pd.read_parquet(ti.xcom_pull(task_ids="load_reference", key="ref_path"))
    live_df = pd.read_parquet(
        ti.xcom_pull(task_ids="fetch_live_window", key="live_path")
    )
    version = ti.xcom_pull(task_ids="load_reference", key="model_version")
    w_start = ti.xcom_pull(task_ids="fetch_live_window", key="window_start")
    w_end = ti.xcom_pull(task_ids="fetch_live_window", key="window_end")

    # Extract only feature columns — columns present in ref that aren't log metadata
    feature_cols = [c for c in ref_df.columns if c not in _LOG_META_COLS]
    live_feature_cols = [c for c in feature_cols if c in live_df.columns]

    if not live_feature_cols:
        raise ValueError(
            "No feature columns found in live window that match the reference dataset. "
            "Check that prediction logs include raw feature values."
        )

    report = compute_drift_report(
        reference_df=ref_df[feature_cols],
        current_df=live_df[live_feature_cols],
        model_version=version,
        window_start=w_start,
        window_end=w_end,
        ref_logs=ref_df,
        cur_logs=live_df,
        save=True,
    )

    ti.xcom_push(key="drift_detected", value=report.overall_drifted)
    ti.xcom_push(key="drifted_features", value=report.drifted_features)
    ti.xcom_push(key="drift_report_id", value=report.report_id)
    ti.xcom_push(key="critical_count", value=report.summary.get("critical_count", 0))
    log.info(
        "Drift: overall_drifted=%s  drifted_features=%s  critical=%d",
        report.overall_drifted,
        report.drifted_features,
        report.summary.get("critical_count", 0),
    )


def task_compute_quality(**ctx):
    """
    Checks data quality on the live prediction window:
    missing rates, out-of-range values, unknown categories.
    """
    import pandas as pd
    from monitoring.quality_report import compute_quality_report  # noqa: PLC0415

    ti = ctx["ti"]
    live_df = pd.read_parquet(
        ti.xcom_pull(task_ids="fetch_live_window", key="live_path")
    )
    version = ti.xcom_pull(task_ids="load_reference", key="model_version")
    w_start = ti.xcom_pull(task_ids="fetch_live_window", key="window_start")
    w_end = ti.xcom_pull(task_ids="fetch_live_window", key="window_end")

    report = compute_quality_report(
        log_df=live_df,
        model_version=version,
        window_start=w_start,
        window_end=w_end,
        save=True,
    )

    ti.xcom_push(key="quality_passed", value=report.overall_passed)
    ti.xcom_push(key="hard_failures", value=report.hard_failures)
    ti.xcom_push(key="quality_report_id", value=report.report_id)
    log.info(
        "Quality: overall_passed=%s  hard_failures=%s",
        report.overall_passed,
        report.hard_failures,
    )


def task_gate(**ctx):
    """
    Aggregates drift + quality outcomes and writes the retrain signal to an
    Airflow Variable. retrain_trigger_dag polls this variable 30 min later.
    """
    ti = ctx["ti"]
    drift_detected = (
        ti.xcom_pull(task_ids="compute_drift", key="drift_detected") or False
    )
    quality_passed = ti.xcom_pull(task_ids="compute_quality", key="quality_passed")
    critical_count = ti.xcom_pull(task_ids="compute_drift", key="critical_count") or 0

    should_retrain = drift_detected or (quality_passed is False)

    Variable.set(RETRAIN_VAR_KEY, str(should_retrain).lower())
    Variable.set("drift_critical_count", str(critical_count))
    Variable.set("last_drift_check_ts", ctx["ts"])

    ti.xcom_push(key="should_retrain", value=should_retrain)

    log.info(
        "Gate: drift_detected=%s  quality_passed=%s  critical=%d  → should_retrain=%s",
        drift_detected,
        quality_passed,
        critical_count,
        should_retrain,
    )

    if should_retrain:
        _send_drift_alert(ctx, drift_detected, quality_passed, critical_count)


def _send_drift_alert(ctx, drift_detected, quality_passed, critical_count):
    try:
        from alerting.notify import send_alert  # noqa: PLC0415

        ti = ctx["ti"]
        lines = []
        if drift_detected:
            features = (
                ti.xcom_pull(task_ids="compute_drift", key="drifted_features") or []
            )
            lines.append(
                f"Feature drift detected: {features}  (critical={critical_count})"
            )
        if not quality_passed:
            failures = (
                ti.xcom_pull(task_ids="compute_quality", key="hard_failures") or []
            )
            lines.append(f"Quality hard failures: {failures}")

        send_alert(
            channel="slack",
            title="[DRIFT CHECK] Retraining triggered",
            message="\n".join(lines),
            severity="critical" if critical_count > 0 else "warning",
            labels={"dag": "ml_drift_check", "ds": ctx["ds"]},
        )
    except Exception as e:
        log.error("Drift alert failed: %s", e)


def _on_failure(context):
    try:
        from alerting.notify import send_alert  # noqa: PLC0415

        ti = context["task_instance"]
        send_alert(
            channel="slack",
            title=f"[DRIFT CHECK FAIL] {ti.task_id}",
            message=str(context.get("exception", "unknown")),
            severity="critical",
            labels={"dag": "ml_drift_check", "task": ti.task_id},
        )
    except Exception as e:
        log.error("Failure alert send failed: %s", e)


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ml_drift_check",
    description="Periodic drift + quality check on live prediction window",
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "monitoring", "drift"],
    default_args={**DEFAULT_ARGS, "on_failure_callback": _on_failure},
    doc_md=__doc__,
) as dag:

    t_ref = PythonOperator(
        task_id="load_reference", python_callable=task_load_reference
    )
    t_live = PythonOperator(
        task_id="fetch_live_window", python_callable=task_fetch_live_window
    )
    t_drift = PythonOperator(
        task_id="compute_drift", python_callable=task_compute_drift
    )
    t_quality = PythonOperator(
        task_id="compute_quality", python_callable=task_compute_quality
    )
    t_gate = PythonOperator(task_id="gate", python_callable=task_gate)

    # drift and quality run in parallel after live window is ready
    t_ref >> t_live >> [t_drift, t_quality] >> t_gate
