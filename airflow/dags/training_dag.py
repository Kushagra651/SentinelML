"""
airflow/dags/training_dag.py

Full ML training pipeline:
  ingest → validate → train → evaluate → register

Schedule : TRAINING_SCHEDULE env (default: weekly Sunday 02:00 UTC)
           Also triggered on-demand by retrain_trigger_dag.

Design notes:
  - train.py handles ingest + feature engineering internally (run_ingestion_pipeline
    + build_features) so the DAG does NOT need separate ingest/features tasks.
  - evaluate.py signature: evaluate(version_tag) — reads artifacts from MODEL_DIR.
  - register_model.py signature: register_model(version_tag, force=False).
  - XCom passes version_tag between tasks — all artifact paths are derived from it.
  - evaluate.py's promotion gate already decides pass/fail; task_evaluate raises if
    the gate fails so the DAG fails fast rather than registering a bad model.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

SCHEDULE = os.getenv("TRAINING_SCHEDULE", "0 2 * * 0")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


# ── Task callables ────────────────────────────────────────────────────────────


def task_train(**ctx):
    """
    Runs the full training pipeline: ingest → validate → build_features → train.
    run_training() handles all of this internally and returns artifact paths.
    """
    from training.train import run_training

    result = run_training()
    ctx["ti"].xcom_push(key="version_tag", value=result["version_tag"])
    ctx["ti"].xcom_push(key="val_accuracy", value=result["val_accuracy"])
    log.info(
        "Training complete. version_tag=%s  val_accuracy=%.4f",
        result["version_tag"],
        result["val_accuracy"],
    )


def task_evaluate(**ctx):
    """
    Runs full held-out test set evaluation.
    Raises ValueError if the promotion gate fails — DAG stops here, no bad model registered.
    """
    from training.evaluate import evaluate

    ti = ctx["ti"]
    version_tag = ti.xcom_pull(task_ids="train", key="version_tag")

    report = evaluate(version_tag=version_tag)

    if not report.get("promote", False):
        metrics = report.get("metrics", {})
        raise ValueError(
            f"Promotion gate failed for v{version_tag}. "
            f"acc={metrics.get('accuracy', 0):.4f}  "
            f"f1={metrics.get('f1', 0):.4f}  "
            f"roc_auc={metrics.get('roc_auc', 0):.4f}"
        )

    metrics = report.get("metrics", {})
    ti.xcom_push(key="promote", value=report["promote"])
    log.info(
        "Evaluation passed. version_tag=%s  f1=%.4f  roc_auc=%.4f",
        version_tag,
        metrics.get("f1", 0),
        metrics.get("roc_auc", 0),
    )


def task_register(**ctx):
    """
    Registers the evaluated model. register_model() does its own comparison
    against the current production model — it may decide not to promote even
    if evaluate.py passed (e.g. not enough improvement over current production).
    """
    from training.register_model import register_model

    ti = ctx["ti"]
    version_tag = ti.xcom_pull(task_ids="train", key="version_tag")

    result = register_model(version_tag=version_tag, force=False)

    ti.xcom_push(key="promoted", value=result["promoted"])
    ti.xcom_push(key="reason", value=result["reason"])
    log.info(
        "Registration complete. version_tag=%s  promoted=%s  reason=%s",
        version_tag,
        result["promoted"],
        result["reason"],
    )


def _on_failure(context):
    try:
        from alerting.notify import send_alert

        ti = context["task_instance"]
        send_alert(
            channel="slack",
            title=f"[TRAINING FAIL] {context['dag'].dag_id} / {ti.task_id}",
            message=str(context.get("exception", "unknown error")),
            severity="critical",
            labels={"dag": context["dag"].dag_id, "task": ti.task_id},
        )
    except Exception as e:
        log.error("Alert send failed: %s", e)


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ml_training_pipeline",
    description="train → evaluate → register_model",
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training"],
    default_args={**DEFAULT_ARGS, "on_failure_callback": _on_failure},
    doc_md=__doc__,
) as dag:

    t_train = PythonOperator(task_id="train", python_callable=task_train)
    t_evaluate = PythonOperator(task_id="evaluate", python_callable=task_evaluate)
    t_register = PythonOperator(task_id="register", python_callable=task_register)

    t_train >> t_evaluate >> t_register
