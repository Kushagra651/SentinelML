# =============================================================================
# training/register_model.py
# =============================================================================
# PURPOSE:
#   Final promotion gate before a model serves production traffic.
#   Reads evaluate.py's report, decides whether to promote, and writes the
#   contract files that predict.py consumes on startup / hot-reload.
#
# CONTRACT FILES WRITTEN:
#   artifacts/production_model.json   ← predict.py reads THIS on startup
#   artifacts/models/model_registry.json  ← full audit history of all versions
#   artifacts/models/promotion_audit.log  ← append-only human log
#
# CRITICAL PATH CONTRACT (must match predict.py exactly):
#   production_model.json schema:
#     {
#       "version":       "v{tag}",
#       "alias":         "production",
#       "model_path":    "/app/artifacts/models/model_v{tag}.pkl",
#       "pipeline_path": "/app/artifacts/models/pipeline_v{tag}.pkl",
#       "registered_at": "<iso timestamp>",
#       "status":        "production",
#       "val_accuracy":  <float from eval report>,
#       "version_tag":   "{tag}"
#     }
#
# WHO CALLS THIS:
#   - airflow/dags/training_dag.py  (after evaluate.py passes)
#   - CLI: python -m training.register_model --version_tag 20240415_143022
#   - CLI rollback: python -m training.register_model --action rollback --version_tag <old>
# =============================================================================

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training.register_model")


# =============================================================================
# PATHS
# =============================================================================
# ARTIFACTS_DIR is the root artifacts directory — same env var used by predict.py
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))

# MODEL_DIR is where .pkl and .json training artifacts live
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/models"))

# THIS IS THE CONTRACT FILE predict.py reads — must be at artifacts/, NOT artifacts/models/
PRODUCTION_FILE = ARTIFACTS_DIR / "production_model.json"

# Registry + audit live inside MODEL_DIR (internal, not read by serving layer)
REGISTRY_FILE = MODEL_DIR / "model_registry.json"
AUDIT_LOG_FILE = MODEL_DIR / "promotion_audit.log"

# Minimum ROC-AUC improvement required to replace the current production model.
# Prevents deploying noise as "better".
MIN_IMPROVEMENT_DELTA = float(os.getenv("MIN_IMPROVEMENT_DELTA", "0.001"))


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        logger.info("No existing registry — starting fresh")
        return {"models": {}, "production_version": None}
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    logger.info(
        "Registry loaded — %d versions, production: %s",
        len(registry.get("models", {})),
        registry.get("production_version", "none"),
    )
    return registry


def _save_registry(registry: dict) -> None:
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    logger.info("Registry saved → %s", REGISTRY_FILE)


def _load_eval_report(version_tag: str) -> dict:
    path = MODEL_DIR / f"eval_report_v{version_tag}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found: {path}\n"
            "Run evaluate.py before registering."
        )
    with open(path) as f:
        report = json.load(f)
    logger.info("Eval report loaded ← %s", path)
    return report


def _load_production_info() -> Optional[dict]:
    """Returns current production_model.json contents, or None if not yet set."""
    if not PRODUCTION_FILE.exists():
        logger.info("No production model currently registered")
        return None
    with open(PRODUCTION_FILE) as f:
        info = json.load(f)
    logger.info(
        "Current production: version_tag=%s  registered_at=%s",
        info.get("version_tag"),
        info.get("registered_at"),
    )
    return info


def _write_production_file(version_tag: str, eval_report: dict) -> None:
    """
    Writes artifacts/production_model.json in the exact schema predict.py expects.

    Schema is fixed — never add/rename keys without updating predict.py too.
    val_accuracy sourced from eval report metrics (accuracy on held-out test set).
    """
    metrics = eval_report.get("metrics", {})
    # Use absolute container paths so predict.py works inside Docker
    model_path = str(MODEL_DIR / f"model_v{version_tag}.pkl")
    pipeline_path = str(MODEL_DIR / f"pipeline_v{version_tag}.pkl")

    # Verify the artifact files actually exist before writing the pointer
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model artifact missing: {model_path}\n"
            "Cannot write production pointer to a file that doesn't exist."
        )
    if not Path(pipeline_path).exists():
        raise FileNotFoundError(
            f"Pipeline artifact missing: {pipeline_path}\n"
            "Cannot write production pointer to a file that doesn't exist."
        )

    production_info = {
        "version": f"v{version_tag}",
        "alias": "production",
        "model_path": model_path,
        "pipeline_path": pipeline_path,
        "registered_at": datetime.utcnow().isoformat(),
        "status": "production",
        "val_accuracy": metrics.get("accuracy", 0.0),
        "version_tag": version_tag,
    }

    with open(PRODUCTION_FILE, "w") as f:
        json.dump(production_info, f, indent=2)

    logger.info("production_model.json written → %s", PRODUCTION_FILE)
    logger.info(
        "  version_tag=%s  model_path=%s  val_accuracy=%.4f",
        version_tag,
        model_path,
        production_info["val_accuracy"],
    )


def _write_audit_entry(
    action: str, version_tag: str, reason: str, metrics: Optional[dict] = None
) -> None:
    """Append-only audit trail. Never truncated or overwritten."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    metrics_str = ""
    if metrics:
        metrics_str = (
            f"acc={metrics.get('accuracy', 'n/a'):.4f}  "
            f"f1={metrics.get('f1', 'n/a'):.4f}  "
            f"roc_auc={metrics.get('roc_auc', 'n/a'):.4f}"
        )
    entry = f"{timestamp} | {action:<14} | v{version_tag} | {metrics_str} | {reason}\n"
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(entry)
    logger.info("Audit: %s", entry.strip())


def _record_in_registry(
    registry: dict,
    version_tag: str,
    status: str,
    eval_report: dict,
    reason: str,
) -> None:
    """
    Upserts a model entry into the registry dict IN PLACE.
    Caller is responsible for calling _save_registry() afterwards.
    """
    registry["models"][version_tag] = {
        "version_tag": version_tag,
        "status": status,
        "recorded_at": datetime.utcnow().isoformat(),
        "reason": reason,
        "metrics": eval_report.get("metrics", {}),
        "promote_flag": eval_report.get("promote"),
        "promotion_gate": eval_report.get("promotion_gate", {}),
        "training_context": eval_report.get("training_context", {}),
        "model_path": str(MODEL_DIR / f"model_v{version_tag}.pkl"),
        "pipeline_path": str(MODEL_DIR / f"pipeline_v{version_tag}.pkl"),
        "eval_report_path": str(MODEL_DIR / f"eval_report_v{version_tag}.json"),
    }


def _archive_current_production(registry: dict) -> None:
    """Marks the current production model as archived in-place. Never deletes artifacts."""
    current = registry.get("production_version")
    if current and current in registry["models"]:
        registry["models"][current]["status"] = "archived"
        registry["models"][current]["archived_at"] = datetime.utcnow().isoformat()
        logger.info("Archived previous production model: v%s", current)


# =============================================================================
# COMPARISON LOGIC
# =============================================================================


def _is_better_than_production(
    new_metrics: dict, prod_info: Optional[dict]
) -> tuple[bool, str]:
    """
    Compares new model against current production.

    Rule 1: No production model → always promote (first deployment).
    Rule 2: New ROC-AUC must exceed current by at least MIN_IMPROVEMENT_DELTA.

    ROC-AUC chosen as primary comparison metric because it is threshold-independent
    and robust to class imbalance — better signal than accuracy on the Adult dataset.
    """
    if prod_info is None:
        return True, "No existing production model — first deployment"

    # Read production metrics from registry (more reliable than production_model.json
    # which only stores val_accuracy, not the full metrics dict)
    registry = _load_registry()
    prod_version = prod_info.get("version_tag")
    prod_metrics = {}
    if prod_version and prod_version in registry.get("models", {}):
        prod_metrics = registry["models"][prod_version].get("metrics", {})

    new_auc = new_metrics.get("roc_auc", 0.0)
    prod_auc = prod_metrics.get("roc_auc", 0.0)
    delta = new_auc - prod_auc

    logger.info(
        "Comparison — new ROC-AUC: %.4f  prod ROC-AUC: %.4f  delta: %+.4f  (min: %+.4f)",
        new_auc, prod_auc, delta, MIN_IMPROVEMENT_DELTA,
    )

    if delta >= MIN_IMPROVEMENT_DELTA:
        return True, (
            f"New ROC-AUC {new_auc:.4f} beats production {prod_auc:.4f} "
            f"by {delta:+.4f} (min required: {MIN_IMPROVEMENT_DELTA:+.4f})"
        )
    elif delta >= 0:
        return False, (
            f"Improvement {delta:+.4f} is below minimum delta {MIN_IMPROVEMENT_DELTA:+.4f}. "
            "Not promoting to avoid deploying noise."
        )
    else:
        return False, (
            f"New ROC-AUC {new_auc:.4f} is worse than production {prod_auc:.4f} "
            f"(delta: {delta:+.4f}). Keeping current production model."
        )


# =============================================================================
# PUBLIC API
# =============================================================================


def register_model(version_tag: str, force: bool = False) -> dict:
    """
    Attempts to promote a trained + evaluated model to production.

    Steps:
      1. Load eval report — raises if evaluate.py hasn't run for this version
      2. Check evaluate.py's promotion gate (metric thresholds)
      3. Compare against current production model (ROC-AUC delta)
      4. If promoting: archive old, write production_model.json, update registry
      5. Write audit log entry for every outcome (promoted / rejected / failed)

    Args:
        version_tag: e.g. '20240415_143022'
        force: Skip comparison check. Use only for hotfixes / manual overrides.

    Returns:
        dict with keys: promoted (bool), version_tag, reason, metrics
    """
    _ensure_dirs()

    logger.info("=" * 60)
    logger.info("Registration started  |  version: %s  |  force=%s", version_tag, force)
    logger.info("=" * 60)

    # ── Step 1: Load eval report ──────────────────────────────────────────────
    eval_report = _load_eval_report(version_tag)
    new_metrics = eval_report.get("metrics", {})

    # ── Step 2: Check evaluate.py's promotion gate ────────────────────────────
    eval_promote = eval_report.get("promote", False)

    if not eval_promote and not force:
        reason = (
            f"evaluate.py promotion gate FAILED — "
            f"acc={new_metrics.get('accuracy', 0):.4f}  "
            f"f1={new_metrics.get('f1', 0):.4f}  "
            f"roc_auc={new_metrics.get('roc_auc', 0):.4f}. "
            "Model does not meet minimum quality thresholds."
        )
        logger.warning("❌  Registration REJECTED: %s", reason)
        registry = _load_registry()
        _record_in_registry(registry, version_tag, "failed", eval_report, reason)
        _save_registry(registry)
        _write_audit_entry("REJECTED", version_tag, reason, new_metrics)
        return {"promoted": False, "version_tag": version_tag, "reason": reason, "metrics": new_metrics}

    if force and not eval_promote:
        logger.warning("⚠️  force=True — bypassing promotion gate (model did NOT pass thresholds)")

    # ── Step 3: Compare against current production ────────────────────────────
    registry = _load_registry()
    prod_info = _load_production_info()

    if force:
        should_promote = True
        reason = "Force promotion — comparison check skipped (force=True)"
        logger.warning("⚠️  %s", reason)
    else:
        should_promote, reason = _is_better_than_production(new_metrics, prod_info)

    # ── Step 4: Decision ──────────────────────────────────────────────────────
    if not should_promote:
        logger.warning("❌  NOT promoted: %s", reason)
        _record_in_registry(registry, version_tag, "not_promoted", eval_report, reason)
        _save_registry(registry)
        _write_audit_entry("NOT_PROMOTED", version_tag, reason, new_metrics)
        return {"promoted": False, "version_tag": version_tag, "reason": reason, "metrics": new_metrics}

    # ── Step 5: Promote ───────────────────────────────────────────────────────
    logger.info("✅  Promoting v%s to production …", version_tag)

    # Archive old production in registry before overwriting pointer
    _archive_current_production(registry)

    # Write the contract file predict.py reads — verify artifacts exist first
    _write_production_file(version_tag, eval_report)

    # Update registry
    _record_in_registry(registry, version_tag, "production", eval_report, reason)
    registry["production_version"] = version_tag
    registry["last_promoted_at"] = datetime.utcnow().isoformat()
    _save_registry(registry)

    _write_audit_entry("PROMOTED", version_tag, reason, new_metrics)

    logger.info("=" * 60)
    logger.info(
        "✅  v%s is now PRODUCTION  |  acc=%.4f  f1=%.4f  roc_auc=%.4f",
        version_tag,
        new_metrics.get("accuracy", 0),
        new_metrics.get("f1", 0),
        new_metrics.get("roc_auc", 0),
    )
    logger.info("=" * 60)

    return {"promoted": True, "version_tag": version_tag, "reason": reason, "metrics": new_metrics}


def rollback(target_version_tag: str) -> dict:
    """
    Restores a previously archived model version to production.

    Does NOT re-run evaluate.py — the target version was already evaluated.
    Verifies .pkl artifacts exist on disk before writing the production pointer.

    Args:
        target_version_tag: Version to restore, e.g. '20240101_090000'

    Returns:
        dict with promoted=True and rollback context
    """
    _ensure_dirs()

    logger.info("=" * 60)
    logger.info("ROLLBACK requested → v%s", target_version_tag)
    logger.info("=" * 60)

    registry = _load_registry()

    if target_version_tag not in registry.get("models", {}):
        raise ValueError(
            f"v{target_version_tag} not found in registry. "
            "Cannot roll back to an unregistered version."
        )

    # Verify artifacts are on disk
    model_path = MODEL_DIR / f"model_v{target_version_tag}.pkl"
    pipeline_path = MODEL_DIR / f"pipeline_v{target_version_tag}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact missing for rollback: {model_path}")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline artifact missing for rollback: {pipeline_path}")

    # Load the eval report for this version to reconstruct production_model.json correctly
    eval_report = _load_eval_report(target_version_tag)

    # Archive current production
    _archive_current_production(registry)

    # Write production_model.json with the target version's artifacts
    _write_production_file(target_version_tag, eval_report)

    # Update registry
    registry["models"][target_version_tag]["status"] = "production"
    registry["models"][target_version_tag]["restored_at"] = datetime.utcnow().isoformat()
    registry["production_version"] = target_version_tag
    registry["last_promoted_at"] = datetime.utcnow().isoformat()
    _save_registry(registry)

    reason = f"Manual rollback to v{target_version_tag}"
    target_metrics = eval_report.get("metrics", {})
    _write_audit_entry("ROLLBACK", target_version_tag, reason, target_metrics)

    logger.info("=" * 60)
    logger.info("✅  Rollback complete — production is now v%s", target_version_tag)
    logger.info("=" * 60)

    return {
        "promoted": True,
        "version_tag": target_version_tag,
        "reason": reason,
        "metrics": target_metrics,
        "is_rollback": True,
    }


def list_models(status_filter: Optional[str] = None) -> list:
    """
    Returns all registered model versions, newest first.

    Args:
        status_filter: 'production' | 'archived' | 'failed' | 'not_promoted' | None (all)
    """
    registry = _load_registry()
    models = list(registry.get("models", {}).values())
    if status_filter:
        models = [m for m in models if m.get("status") == status_filter]
    models.sort(key=lambda m: m.get("recorded_at", ""), reverse=True)
    return models


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
# Promote:   python -m training.register_model --version_tag 20240415_143022
# Force:     python -m training.register_model --version_tag 20240415_143022 --force
# Rollback:  python -m training.register_model --action rollback --version_tag 20240101_090000
# List all:  python -m training.register_model --action list
# List prod: python -m training.register_model --action list --status production
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register or rollback ML model versions.")
    parser.add_argument("--action", type=str, default="promote",
                        choices=["promote", "rollback", "list"])
    parser.add_argument("--version_tag", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", type=str, default=None,
                        choices=["production", "archived", "failed", "not_promoted"])
    args = parser.parse_args()

    if args.action == "promote":
        if not args.version_tag:
            parser.error("--version_tag is required for promote")
        result = register_model(args.version_tag, force=args.force)
        print(f"\n  Promoted : {result['promoted']}")
        print(f"  Version  : {result['version_tag']}")
        print(f"  Reason   : {result['reason']}")
        if result.get("metrics"):
            m = result["metrics"]
            print(f"  Metrics  : acc={m.get('accuracy', 0):.4f}  "
                  f"f1={m.get('f1', 0):.4f}  roc_auc={m.get('roc_auc', 0):.4f}")

    elif args.action == "rollback":
        if not args.version_tag:
            parser.error("--version_tag is required for rollback")
        result = rollback(args.version_tag)
        print(f"\n  Rollback complete → production is now v{result['version_tag']}")

    elif args.action == "list":
        models = list_models(status_filter=args.status)
        print(f"\n{'VERSION TAG':<22} {'STATUS':<15} {'ROC-AUC':<10} {'F1':<10} RECORDED AT")
        print("-" * 80)
        for m in models:
            metrics = m.get("metrics", {})
            roc = metrics.get("roc_auc", "n/a")
            f1 = metrics.get("f1", "n/a")
            roc_str = f"{roc:.4f}" if isinstance(roc, float) else roc
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
            print(f"{m['version_tag']:<22} {m.get('status', 'unknown'):<15} "
                  f"{roc_str:<10} {f1_str:<10} {m.get('recorded_at', '')}")