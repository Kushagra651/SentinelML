# =============================================================================
# training/evaluate.py
# =============================================================================
# PURPOSE:
#   Full model evaluation on a HELD-OUT test set — completely separate from
#   the quick val-split done inside train.py.
#
#   Called by:
#     - airflow/dags/training_dag.py  (runs automatically after train.py)
#     - register_model.py             (reads the report to decide promotion)
#     - CLI: python -m training.evaluate --version_tag 20240415_143022
#
#   Outputs (written next to the model artifact):
#     - eval_report_v{tag}.json   — all metrics in machine-readable form
# =============================================================================
 
import json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional
 
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
 
from data.ingest import load_from_feature_store
from data.validate import validate_dataframe
from data.features import FeaturePipeline, build_features
 
# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training.evaluate")
 
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
 
# =============================================================================
# CONSTANTS
# =============================================================================
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/models"))
TARGET_COL = os.getenv("TARGET_COLUMN", "income")
 
THRESHOLD_ACCURACY = float(os.getenv("EVAL_MIN_ACCURACY", "0.75"))
THRESHOLD_F1 = float(os.getenv("EVAL_MIN_F1", "0.70"))
THRESHOLD_ROC_AUC = float(os.getenv("EVAL_MIN_ROC_AUC", "0.75"))
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
 
 
# =============================================================================
# ARTIFACT LOADERS
# =============================================================================
 
 
def load_model(version_tag: str):
    path = MODEL_DIR / f"model_v{version_tag}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded ← %s", path)
    return model
 
 
def load_pipeline(version_tag: str) -> FeaturePipeline:
    path = MODEL_DIR / f"pipeline_v{version_tag}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline artifact not found: {path}")
    pipeline = FeaturePipeline.load(str(path))
    logger.info("Feature pipeline loaded ← %s", path)
    return pipeline
 
 
def load_train_meta(version_tag: str) -> dict:
    path = MODEL_DIR / f"train_meta_v{version_tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"Training metadata not found: {path}")
    with open(path) as f:
        meta = json.load(f)
    logger.info("Training metadata loaded ← %s", path)
    return meta
 
 
# =============================================================================
# METRIC HELPERS
# =============================================================================
 
 
def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 6),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "positive_rate": round(float(y_true.mean()), 4),
        "decision_threshold": DECISION_THRESHOLD,
    }
 
 
def _confusion_matrix_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    tn, fp, fn, tp = cm_raw.ravel()
    return {
        "raw": cm_raw.tolist(),
        "normalized": np.round(cm_norm, 4).tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0, 6),
        "false_negative_rate": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0, 6),
    }
 
 
def _per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report_str = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    logger.info("Per-class classification report:\n%s", report_str)
    return report_dict
 
 
def _threshold_analysis(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[list] = None
) -> list:
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 1.0, 0.1)]
    results = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "n_predicted_positive": int(preds.sum()),
        })
    return results
 
 
def _feature_importances(model, pipeline: FeaturePipeline, top_n: int = 20) -> list:
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not expose feature_importances_ — skipping")
        return []
    importances = model.feature_importances_
    feature_names = pipeline.feature_names()
 
    if len(feature_names) != len(importances):
        logger.warning(
            "Feature name count (%d) ≠ importance count (%d) — using indices",
            len(feature_names), len(importances),
        )
        feature_names = [f"feature_{i}" for i in range(len(importances))]
 
    ranked = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {"rank": i + 1, "feature": name, "importance": round(imp, 6)}
        for i, (name, imp) in enumerate(ranked[:top_n])
    ]
 
 
def _calibration_check(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))
    return {
        "expected_calibration_error": round(ece, 6),
        "n_bins": n_bins,
        "mean_predicted_value": mean_pred.round(4).tolist(),
        "fraction_of_positives": fraction_pos.round(4).tolist(),
        "calibration_quality": "good" if ece < 0.05 else "fair" if ece < 0.10 else "poor",
    }
 
 
def _check_promotion_thresholds(metrics: dict) -> dict:
    checks = {
        "accuracy": metrics["accuracy"] >= THRESHOLD_ACCURACY,
        "f1": metrics["f1"] >= THRESHOLD_F1,
        "roc_auc": metrics["roc_auc"] >= THRESHOLD_ROC_AUC,
    }
    all_passed = all(checks.values())
    thresholds = {
        "accuracy": THRESHOLD_ACCURACY,
        "f1": THRESHOLD_F1,
        "roc_auc": THRESHOLD_ROC_AUC,
    }
    for metric, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(
            "  Promotion gate [%s]: %.4f  (threshold %.4f)  → %s",
            metric, metrics[metric], thresholds[metric], status,
        )
    if all_passed:
        logger.info("✅  All promotion gates PASSED")
    else:
        logger.warning("❌  One or more promotion gates FAILED")
 
    return {"thresholds_used": thresholds, "checks": checks, "promote": all_passed}
 
 
# =============================================================================
# CORE EVALUATION FUNCTION
# =============================================================================
 
 
def evaluate(version_tag: str, test_data_source: Optional[str] = None) -> dict:
    """
    Runs the full evaluation suite for a trained model version.
 
    Args:
        version_tag: Which model version to evaluate (e.g. '20240415_143022')
        test_data_source: Unused — kept for API compatibility with Airflow DAG.
 
    Returns:
        Full evaluation report as a dict (also saved as JSON to MODEL_DIR)
    """
    logger.info("=" * 60)
    logger.info("Evaluation started  |  version: %s", version_tag)
    logger.info("=" * 60)
 
    # ── Step 1: Load artifacts ────────────────────────────────────────────────
    logger.info("Step 1/6 — Loading model artifacts …")
    model = load_model(version_tag)
    pipeline = load_pipeline(version_tag)
    train_meta = load_train_meta(version_tag)
 
    # ── Step 2: Load + validate test data ─────────────────────────────────────
    logger.info("Step 2/6 — Loading test data …")
    raw_test = load_from_feature_store(split="test")
 
    report = validate_dataframe(raw_test)
    if not report.passed:
        raise ValueError("Test data failed hard validation checks — cannot evaluate.")
 
    logger.info("Test data loaded: %d rows", len(raw_test))
 
    # ── Step 3: Feature transform (NO fit) ────────────────────────────────────
    logger.info("Step 3/6 — Transforming test features …")
    X_test, y_test_raw, _ = build_features(raw_test, fit=False, pipeline=pipeline)
    logger.info("Test feature matrix: %s", X_test.shape)
 
    # ── Step 4: Generate predictions ──────────────────────────────────────────
    logger.info("Step 4/6 — Generating predictions …")
 
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
 
    # Convert string labels to binary (">50K" → 1, "<=50K" → 0)
    y_test = (y_test_raw == ">50K").astype(int).values
 
    # ── Step 5: Compute metrics ───────────────────────────────────────────────
    logger.info("Step 5/6 — Computing metrics …")
 
    core_metrics = _classification_metrics(y_test, y_pred, y_prob)
    conf_matrix = _confusion_matrix_block(y_test, y_pred)
    per_class = _per_class_report(y_test, y_pred)
    thresholds = _threshold_analysis(y_test, y_prob)
    importances = _feature_importances(model, pipeline, top_n=20)
    calibration = _calibration_check(y_test, y_prob)
    promotion_gate = _check_promotion_thresholds(core_metrics)
 
    # ── Step 6: Assemble + save report ────────────────────────────────────────
    logger.info("Step 6/6 — Saving evaluation report …")
 
    report_dict = {
        "version_tag": version_tag,
        "evaluated_at": pd.Timestamp.utcnow().isoformat(),
        "promote": promotion_gate["promote"],
        "metrics": core_metrics,
        "confusion_matrix": conf_matrix,
        "per_class_report": per_class,
        "threshold_analysis": thresholds,
        "feature_importances": importances,
        "calibration": calibration,
        "promotion_gate": promotion_gate,
        "training_context": {
            "n_train": train_meta.get("n_train"),
            "n_val": train_meta.get("n_val"),
            "val_accuracy": train_meta.get("val_accuracy"),
            "hyperparameters": train_meta.get("hyperparameters"),
            "trained_at": train_meta.get("trained_at"),
        },
    }
 
    report_path = MODEL_DIR / f"eval_report_v{version_tag}.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
 
    logger.info("Evaluation report saved → %s", report_path)
    logger.info("=" * 60)
    logger.info(
        "Evaluation COMPLETE  |  accuracy=%.4f  f1=%.4f  roc_auc=%.4f  promote=%s",
        core_metrics["accuracy"], core_metrics["f1"],
        core_metrics["roc_auc"], promotion_gate["promote"],
    )
    logger.info("=" * 60)
 
    return report_dict
 
 
# =============================================================================
# CLI ENTRY POINT
# =============================================================================
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Evaluate a trained model version.")
    parser.add_argument("--version_tag", type=str, required=True)
    parser.add_argument("--test_data_source", type=str, default=None)
    args = parser.parse_args()
 
    result = evaluate(
        version_tag=args.version_tag,
        test_data_source=args.test_data_source,
    )
 
    print("\n--- Evaluation Summary ---")
    print(f"  Accuracy : {result['metrics']['accuracy']:.4f}")
    print(f"  F1 Score : {result['metrics']['f1']:.4f}")
    print(f"  ROC-AUC  : {result['metrics']['roc_auc']:.4f}")
    print(f"  Promote  : {'YES ✅' if result['promote'] else 'NO ❌'}")