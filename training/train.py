# =============================================================================
# training/train.py
# =============================================================================
# PURPOSE:
#   Central training script. Orchestrates the full model training pipeline:
#     1. Pull raw data via ingest.py
#     2. Validate it via validate.py
#     3. Build features via features.py
#     4. Train a GradientBoostingClassifier
#     5. Save the trained model + feature pipeline as versioned artifacts
#
#   Called by:
#     - airflow/dags/training_dag.py  (scheduled / triggered retraining)
#     - CLI: python -m training.train
#
#   Outputs (written to MODEL_DIR):
#     - model_v{timestamp}.pkl        — trained sklearn model
#     - pipeline_v{timestamp}.pkl     — fitted FeaturePipeline
#     - train_meta_v{timestamp}.json  — run metadata (params, data stats, etc.)
# =============================================================================

import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from data.ingest import load_from_feature_store, run_ingestion_pipeline
from data.validate import validate_dataframe
from data.features import build_features, FeaturePipeline

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training.train")

# =============================================================================
# CONSTANTS
# =============================================================================
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/models"))
TARGET_COL = "income"
VALIDATION_SPLIT = float(os.getenv("TRAIN_VAL_SPLIT", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

DEFAULT_PARAMS = {
    "n_estimators": int(os.getenv("GBM_N_ESTIMATORS", "200")),
    "max_depth": int(os.getenv("GBM_MAX_DEPTH", "4")),
    "learning_rate": float(os.getenv("GBM_LEARNING_RATE", "0.05")),
    "subsample": float(os.getenv("GBM_SUBSAMPLE", "0.8")),
    "random_state": RANDOM_SEED,
}


# =============================================================================
# ARTIFACT HELPERS
# =============================================================================


def _make_version_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Model artifact directory: %s", MODEL_DIR.resolve())


def save_model(model, version_tag: str) -> Path:
    path = MODEL_DIR / f"model_v{version_tag}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", path)
    return path


def save_pipeline(pipeline: FeaturePipeline, version_tag: str) -> Path:
    path = MODEL_DIR / f"pipeline_v{version_tag}.pkl"
    pipeline.save(str(path))
    logger.info("Feature pipeline saved → %s", path)
    return path


def save_metadata(meta: dict, version_tag: str) -> Path:
    path = MODEL_DIR / f"train_meta_v{version_tag}.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Training metadata saved → %s", path)
    return path


# =============================================================================
# CORE TRAINING LOGIC
# =============================================================================


def run_training(
    model_params: Optional[dict] = None,
    data_source: Optional[str] = None,
) -> dict:
    """
    End-to-end training run. Called by the Airflow DAG or CLI.

    Returns dict with version_tag, model_path, pipeline_path,
    meta_path, val_accuracy, n_train, n_val.
    """
    version_tag = _make_version_tag()
    _ensure_model_dir()
    params = {**DEFAULT_PARAMS, **(model_params or {})}

    logger.info("=" * 60)
    logger.info("Training run started  |  version: %s", version_tag)
    logger.info("Hyperparameters: %s", params)

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    logger.info("Step 1/7 — Ingesting data …")
    t0 = time.perf_counter()

    run_ingestion_pipeline()
    # Use load_from_feature_store — works regardless of working directory
    raw_df = load_from_feature_store("train")

    logger.info(
        "Loaded %d rows × %d cols in %.2fs",
        len(raw_df), raw_df.shape[1], time.perf_counter() - t0,
    )

    # ── Step 2: Validate ──────────────────────────────────────────────────────
    logger.info("Step 2/7 — Validating data …")
    report = validate_dataframe(raw_df)

    if not report.passed:
        for check in report.errors:
            logger.error("  HARD CHECK FAILED: %s — %s", check.check, check.detail)
        raise ValueError(
            f"Data validation failed ({len(report.errors)} hard checks). "
            "Aborting training."
        )

    for warning in report.warnings:
        logger.warning("  Soft check warning: %s", warning)

    logger.info("Data validation passed (%d rows clean)", report.row_count)

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    logger.info("Step 3/7 — Building features (fit=True) …")
    X, y, pipeline = build_features(raw_df, fit=True)

    logger.info("Feature matrix shape: %s", X.shape)

    # ── Step 4: Train/val split ───────────────────────────────────────────────
    logger.info("Step 4/7 — Splitting data (val=%.0f%%) …", VALIDATION_SPLIT * 100)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    logger.info("Split sizes — train: %d  |  val: %d", len(y_train), len(y_val))

    # ── Step 5: Train model ───────────────────────────────────────────────────
    logger.info("Step 5/7 — Training model …")
    t1 = time.perf_counter()

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    train_duration = time.perf_counter() - t1
    logger.info("Model trained in %.2fs", train_duration)

    # ── Step 6: Quick val check ───────────────────────────────────────────────
    logger.info("Step 6/7 — Evaluating on val split …")

    val_preds = model.predict(X_val)
    val_accuracy = float(np.mean(val_preds == y_val))
    logger.info("Val accuracy: %.4f", val_accuracy)

    MIN_ACCEPTABLE_ACCURACY = float(os.getenv("MIN_TRAIN_ACCURACY", "0.5"))
    if val_accuracy < MIN_ACCEPTABLE_ACCURACY:
        raise ValueError(
            f"Val accuracy {val_accuracy:.4f} < minimum {MIN_ACCEPTABLE_ACCURACY}. "
            "Aborting save."
        )

    # ── Step 7: Save artifacts ────────────────────────────────────────────────
    logger.info("Step 7/7 — Saving artifacts …")

    model_path = save_model(model, version_tag)
    pipeline_path = save_pipeline(pipeline, version_tag)

    meta = {
        "version_tag": version_tag,
        "trained_at": datetime.utcnow().isoformat(),
        "n_rows_ingested": len(raw_df),
        "n_rows_validated": report.row_count,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(X.shape[1]),
        "val_accuracy": round(val_accuracy, 6),
        "train_duration_s": round(train_duration, 3),
        "hyperparameters": params,
        "model_path": str(model_path),
        "pipeline_path": str(pipeline_path),
        "target_column": TARGET_COL,
        "random_seed": RANDOM_SEED,
        "soft_check_warnings": [{"message": w} for w in report.warnings],
    }

    meta_path = save_metadata(meta, version_tag)

    logger.info("=" * 60)
    logger.info(
        "Training COMPLETE  |  version: %s  |  val_acc: %.4f",
        version_tag, val_accuracy,
    )
    logger.info("=" * 60)

    return {
        "version_tag": version_tag,
        "model_path": str(model_path),
        "pipeline_path": str(pipeline_path),
        "meta_path": str(meta_path),
        "val_accuracy": val_accuracy,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }


# Alias used by Airflow DAG
train = run_training


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the ML model from scratch.")
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--data_source", type=str, default=None)
    args = parser.parse_args()

    overrides = {
        k: v for k, v in vars(args).items()
        if v is not None and k != "data_source"
    }

    result = run_training(model_params=overrides or None, data_source=args.data_source)

    print("\n--- Training Result ---")
    for k, v in result.items():
        print(f"  {k}: {v}")