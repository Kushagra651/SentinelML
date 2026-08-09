"""
airflow/dags/retrain_trigger_dag.py

Reads the `drift_retrain_triggered` Airflow Variable set by drift_check_dag's
gate task, then conditionally triggers ml_training_pipeline.

Schedule : 30 min after drift_check (RETRAIN_TRIGGER_SCHEDULE env, default: "30 */6 * * *")

Guard rails:
  - Cooldown: RETRAIN_COOLDOWN_HOURS (default 12h) between automated retrains
  - Daily cap: RETRAIN_MAX_PER_DAY (default 2) triggered retrains per day
  - Manual override: set Airflow Variable FORCE_RETRAIN=true

Design notes:
  - Reads the Airflow Variable written by drift_check_dag gate task, not XCom
    from a specific task instance — simpler and more reliable across DAG runs.
  - Cooldown timestamps stored in a JSON file under ARTIFACTS_DIR since Airflow
    Variables have no TTL and DagRun.find() API differs across Airflow versions.
  - All datetime comparisons use timezone-aware UTC datetimes to avoid TypeError
    on naive vs aware subtraction.
  - send_alert() uses severity= kwarg (not level=).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

SCHEDULE = os.getenv("RETRAIN_TRIGGER_SCHEDULE", "30 */6 * * *")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
TRAINING_DAG_ID = "ml_training_pipeline"
COOLDOWN_HOURS = int(os.getenv("RETRAIN_COOLDOWN_HOURS", "12"))
MAX_PER_DAY = int(os.getenv("RETRAIN_MAX_PER_DAY", "2"))
COOLDOWN_FILE = Path(ARTIFACTS_DIR) / "retrain_cooldown.json"
RETRAIN_VAR_KEY = "drift_retrain_triggered"
FORCE_RETRAIN_VAR = "FORCE_RETRAIN"

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=15),
}


# ── Cooldown state helpers ────────────────────────────────────────────────────


def _read_cooldown() -> dict:
    if COOLDOWN_FILE.exists():
        try:
            with open(COOLDOWN_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_retrain_utc": None, "runs_today": 0, "today_date": None}


def _write_cooldown(state: dict) -> None:
    COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(COOLDOWN_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _runs_today(state: dict) -> int:
    today = str(datetime.now(timezone.utc).date())
    if state.get("today_date") == today:
        return state.get("runs_today", 0)
    return 0  # date rolled over — reset


# ── Task callables ────────────────────────────────────────────────────────────


def task_poll_signal(**ctx):
    """
    Reads the drift_retrain_triggered Variable set by drift_check_dag.
    Also checks FORCE_RETRAIN Variable for manual overrides.
    Pushes a signal dict to XCom.
    """
    force = False
    try:
        force = Variable.get(FORCE_RETRAIN_VAR, default_var="false").lower() == "true"
        if force:
            log.warning("FORCE_RETRAIN=true — overriding drift signal.")
    except Exception:
        pass

    if force:
        signal = {"should_retrain": True, "source": "manual_override"}
    else:
        try:
            val = Variable.get(RETRAIN_VAR_KEY, default_var="false")
            should_retrain = val.lower() == "true"
        except Exception as e:
            log.error("Could not read %s Variable: %s. Defaulting to false.", RETRAIN_VAR_KEY, e)
            should_retrain = False
        signal = {"should_retrain": should_retrain, "source": "drift_check_variable"}

    ctx["ti"].xcom_push(key="signal", value=signal)
    log.info("Retrain signal: %s", signal)


def task_check_cooldown(**ctx):
    """
    Applies cooldown + daily cap guard rails.
    Pushes approved=True/False to XCom.
    """
    ti = ctx["ti"]
    signal = ti.xcom_pull(task_ids="poll_signal", key="signal") or {}

    if not signal.get("should_retrain"):
        ti.xcom_push(key="cooldown", value={"approved": False, "reason": "no_signal"})
        return

    state = _read_cooldown()
    now = datetime.now(timezone.utc)

    # Cooldown window check
    if state.get("last_retrain_utc"):
        last_str = state["last_retrain_utc"]
        last = datetime.fromisoformat(last_str)
        # Ensure timezone-aware for safe subtraction
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        elapsed_h = (now - last).total_seconds() / 3600
        if elapsed_h < COOLDOWN_HOURS:
            reason = f"cooldown ({elapsed_h:.1f}h elapsed, need {COOLDOWN_HOURS}h)"
            log.info("Retrain suppressed: %s", reason)
            ti.xcom_push(key="cooldown", value={"approved": False, "reason": reason})
            return

    # Daily cap check (from cooldown file — simple, no DagRun query needed)
    today_count = _runs_today(state)
    if today_count >= MAX_PER_DAY:
        reason = f"daily_cap ({today_count}/{MAX_PER_DAY} retrains today)"
        log.info("Retrain suppressed: %s", reason)
        ti.xcom_push(key="cooldown", value={"approved": False, "reason": reason})
        return

    log.info("Retrain approved. today_count=%d/%d", today_count, MAX_PER_DAY)
    ti.xcom_push(key="cooldown", value={"approved": True, "reason": "cleared"})


def task_branch(**ctx):
    cooldown = ctx["ti"].xcom_pull(task_ids="check_cooldown", key="cooldown") or {}
    return "trigger_training" if cooldown.get("approved") else "skip_retrain"


def task_notify_triggered(**ctx):
    """Sends Slack alert that an automated retrain was kicked off."""
    try:
        from alerting.notify import send_alert  # noqa: PLC0415

        ti = ctx["ti"]
        signal = ti.xcom_pull(task_ids="poll_signal", key="signal") or {}

        send_alert(
            channel="slack",
            title="[RETRAIN TRIGGER] Automated retrain initiated",
            message=(
                f"Source: {signal.get('source', 'unknown')}\n"
                f"Training DAG `{TRAINING_DAG_ID}` has been triggered."
            ),
            severity="warning",
            labels={"dag": "ml_retrain_trigger", "ds": ctx["ds"]},
        )
    except Exception as e:
        log.error("Retrain notification failed: %s", e)


def task_update_cooldown(**ctx):
    """Writes updated cooldown state after a successful retrain trigger."""
    now = datetime.now(timezone.utc)
    today = str(now.date())
    state = _read_cooldown()

    if state.get("today_date") != today:
        state["runs_today"] = 0
        state["today_date"] = today

    state["last_retrain_utc"] = now.isoformat()
    state["runs_today"] = state.get("runs_today", 0) + 1
    _write_cooldown(state)
    log.info(
        "Cooldown updated: last=%s  runs_today=%d",
        now.isoformat(), state["runs_today"],
    )


def task_clear_force_var(**ctx):
    """Resets FORCE_RETRAIN Variable so it doesn't trigger indefinitely."""
    try:
        Variable.set(FORCE_RETRAIN_VAR, "false")
        log.info("FORCE_RETRAIN Variable reset to false.")
    except Exception as e:
        log.warning("Could not reset %s Variable: %s", FORCE_RETRAIN_VAR, e)


def task_skip_retrain(**ctx):
    cooldown = ctx["ti"].xcom_pull(task_ids="check_cooldown", key="cooldown") or {}
    signal = ctx["ti"].xcom_pull(task_ids="poll_signal", key="signal") or {}
    log.info(
        "Retrain skipped. should_retrain=%s  reason=%s",
        signal.get("should_retrain"),
        cooldown.get("reason"),
    )


def _on_failure(context):
    try:
        from alerting.notify import send_alert  # noqa: PLC0415

        ti = context["task_instance"]
        send_alert(
            channel="slack",
            title=f"[RETRAIN TRIGGER FAIL] {ti.task_id}",
            message=str(context.get("exception", "unknown")),
            severity="critical",
            labels={"dag": "ml_retrain_trigger", "task": ti.task_id},
        )
    except Exception as e:
        log.error("Failure alert send failed: %s", e)


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="ml_retrain_trigger",
    description="Reads drift signal and conditionally triggers ml_training_pipeline",
    schedule_interval=SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "retrain", "trigger"],
    default_args={**DEFAULT_ARGS, "on_failure_callback": _on_failure},
    doc_md=__doc__,
) as dag:

    t_poll = PythonOperator(task_id="poll_signal", python_callable=task_poll_signal)
    t_cooldown = PythonOperator(task_id="check_cooldown", python_callable=task_check_cooldown)
    t_branch = BranchPythonOperator(task_id="branch", python_callable=task_branch)

    t_trigger = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id=TRAINING_DAG_ID,
        wait_for_completion=False,
        reset_dag_run=False,
        conf={"triggered_by": "ml_retrain_trigger", "reason": "drift_or_quality_failure"},
    )

    t_notify = PythonOperator(task_id="notify_triggered", python_callable=task_notify_triggered)
    t_update_cd = PythonOperator(task_id="update_cooldown", python_callable=task_update_cooldown)
    t_clear = PythonOperator(task_id="clear_force_var", python_callable=task_clear_force_var)
    t_skip = PythonOperator(task_id="skip_retrain", python_callable=task_skip_retrain)

    t_done = EmptyOperator(
        task_id="done",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_poll >> t_cooldown >> t_branch
    t_branch >> [t_trigger, t_skip]
    t_trigger >> t_notify >> t_update_cd >> t_clear >> t_done
    t_skip >> t_done