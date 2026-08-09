"""
api/predict.py
==============
Model loading, hot-reload detection, and prediction engine.

Responsibilities
----------------
- Read artifacts/production_model.json to locate the current production model
- Load model_v<tag>.pkl (GBM) and pipeline_v<tag>.pkl (FeaturePipeline) at startup
- Detect when production_model.json has been updated and hot-reload without restart
- Validate incoming feature dicts against the 14 known UCI Adult Income columns
- Run pipeline.transform → model.predict / predict_proba
- Return a structured PredictionResult with prediction, probabilities, latency, model version

Public API
----------
    predict(features: dict) -> PredictionResult
    predict_batch(records: list[dict]) -> list[PredictionResult]
    get_model_info()        -> dict
    reload_if_stale()       -> bool   # called by main.py on each request
    force_reload()          -> None   # admin endpoint / tests
    _ensure_loaded()        -> _ModelBundle
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths — override via environment variables for Docker / tests
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
REGISTRY_FILE = ARTIFACTS_DIR / "production_model.json"

# ---------------------------------------------------------------------------
# Known feature columns — UCI Adult Income, 14 features
# Source of truth: project_context.md + api/schemas.py
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Structured return value from predict()."""

    prediction: int  # 0 (<=50K) or 1 (>50K)
    probability_class_0: float
    probability_class_1: float
    confidence: float  # max(probability_class_0, probability_class_1)
    model_version: str
    model_alias: str  # e.g. "production"
    latency_ms: float
    features_used: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "probability": {
                "class_0": round(self.probability_class_0, 6),
                "class_1": round(self.probability_class_1, 6),
            },
            "confidence": round(self.confidence, 6),
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "latency_ms": round(self.latency_ms, 3),
            "features_used": self.features_used,
            "warnings": self.warnings,
        }


@dataclass
class _ModelBundle:
    """Internal holder for a loaded model + pipeline snapshot."""

    model: Any
    pipeline: Any
    version: str
    alias: str
    registry_mtime: float  # mtime of production_model.json at load time
    loaded_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Module-level state — thread-safe via RLock
# ---------------------------------------------------------------------------
_bundle: _ModelBundle | None = None
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_registry() -> dict:
    """Parse production_model.json and return its contents."""
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"production_model.json not found at {REGISTRY_FILE}. "
            "Run training/register_model.py first."
        )
    with open(REGISTRY_FILE, "r") as fh:
        return json.load(fh)


def _load_pickle(path: Path) -> Any:
    """Load a pickle file with a descriptive error on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_bundle() -> _ModelBundle:
    """
    Read production_model.json, resolve artifact paths, and load model + pipeline.

    production_model.json shape (written by register_model.py):
    {
      "version":       "v20260418_220902",
      "alias":         "production",
      "model_path":    "/app/artifacts/models/model_v20260418_220902.pkl",
      "pipeline_path": "/app/artifacts/models/pipeline_v20260418_220902.pkl",
      "status":        "production",
      "val_accuracy":  0.863252,
      ...
    }
    """
    registry = _read_registry()

    version = registry.get("version", "unknown")
    alias = registry.get("alias", "production")

    if "model_path" not in registry or "pipeline_path" not in registry:
        raise KeyError(
            "production_model.json is missing 'model_path' or 'pipeline_path'. "
            "Re-run register_model.py."
        )

    model_path = Path(registry["model_path"])
    pipeline_path = Path(registry["pipeline_path"])

    logger.info("Loading model bundle version=%s from %s", version, model_path)
    model = _load_pickle(model_path)
    pipeline = _load_pickle(pipeline_path)

    mtime = REGISTRY_FILE.stat().st_mtime
    return _ModelBundle(
        model=model,
        pipeline=pipeline,
        version=version,
        alias=alias,
        registry_mtime=mtime,
    )


def _ensure_loaded() -> _ModelBundle:
    """Return the current bundle, loading it on the first call (double-checked locking)."""
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                _bundle = _load_bundle()
    return _bundle


# ---------------------------------------------------------------------------
# Public: hot-reload
# ---------------------------------------------------------------------------


def reload_if_stale() -> bool:
    """
    Check whether production_model.json has been modified since the bundle was
    last loaded. If so, atomically swap in a fresh bundle.

    Returns True if a reload happened, False otherwise.
    Called by main.py on every request (cheap — just a stat() call normally).
    """
    global _bundle
    if not REGISTRY_FILE.exists():
        return False

    current_mtime = REGISTRY_FILE.stat().st_mtime
    bundle = _ensure_loaded()

    if current_mtime <= bundle.registry_mtime:
        return False  # nothing changed

    logger.info(
        "production_model.json updated (mtime %.3f → %.3f) — hot-reloading model",
        bundle.registry_mtime,
        current_mtime,
    )
    with _lock:
        if current_mtime > _bundle.registry_mtime:  # re-check inside lock
            try:
                new_bundle = _load_bundle()
                _bundle = new_bundle
                logger.info("Hot-reload complete: version=%s", new_bundle.version)
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error("Hot-reload failed — keeping old bundle. Error: %s", exc)
    return False


def force_reload() -> None:
    """Unconditionally reload the model bundle (used by admin endpoint / tests)."""
    global _bundle
    with _lock:
        _bundle = _load_bundle()
    logger.info("Force-reload complete: version=%s", _bundle.version)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_features(features: dict, warnings: list[str]) -> pd.DataFrame:
    """
    Validate and normalise a raw feature dict before pipeline.transform().

    Checks:
    - All 14 expected columns are present (raises ValueError on missing)
    - Unknown extra columns are dropped with a warning
    - Numerical columns are cast to float64; categorical remain as object

    Returns a single-row DataFrame ready for FeaturePipeline.transform().
    """
    missing = [col for col in ALL_FEATURES if col not in features]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    extra = [col for col in features if col not in ALL_FEATURES]
    if extra:
        warnings.append(f"Ignoring unknown columns: {extra}")

    # Build ordered dict with only the 14 known features
    filtered = {col: features[col] for col in ALL_FEATURES}
    df = pd.DataFrame([filtered])

    # Safe dtype casting for numericals
    for col in NUMERICAL_FEATURES:
        try:
            df[col] = df[col].astype("float64")
        except (ValueError, TypeError) as exc:
            warnings.append(f"Could not cast '{col}' to float64: {exc}")

    return df


# ---------------------------------------------------------------------------
# Public: single prediction
# ---------------------------------------------------------------------------


def predict(features: dict) -> PredictionResult:
    """
    Run end-to-end inference for a single record.

    Parameters
    ----------
    features : dict
        Raw feature key-value pairs matching the 14 UCI Adult Income columns.

    Returns
    -------
    PredictionResult
        Prediction (0/1), probabilities, confidence, model metadata, latency.

    Raises
    ------
    ValueError
        If required feature columns are missing.
    RuntimeError
        If the model artifact cannot be loaded or inference fails.
    """
    t_start = time.perf_counter()
    warnings_list: list[str] = []

    # 1. Ensure model is loaded
    try:
        bundle = _ensure_loaded()
    except Exception as exc:
        raise RuntimeError(f"Model bundle could not be loaded: {exc}") from exc

    # 2. Validate + build DataFrame
    df = _validate_features(features, warnings_list)
    feature_cols = df.columns.tolist()

    # 3. Transform through FeaturePipeline
    try:
        X = bundle.pipeline.transform(df)
    except Exception as exc:
        raise RuntimeError(f"Pipeline transform failed: {exc}") from exc

    # 4. Inference
    try:
        # GBM trained on UCI Adult returns string labels ">50K" / "<=50K"
        raw_prediction = bundle.model.predict(X)[0]
        if isinstance(raw_prediction, str):
            prediction = 1 if raw_prediction.strip() == ">50K" else 0
        else:
            prediction = int(raw_prediction)

        if hasattr(bundle.model, "predict_proba"):
            proba = bundle.model.predict_proba(X)[0]
            if len(proba) >= 2:
                prob_0, prob_1 = float(proba[0]), float(proba[1])
            else:
                prob_1 = float(proba[0])
                prob_0 = 1.0 - prob_1
        else:
            prob_1 = 1.0 if prediction == 1 else 0.0
            prob_0 = 1.0 - prob_1
            warnings_list.append(
                "Model does not support predict_proba; probabilities are hard 0/1"
            )

    except Exception as exc:
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    return PredictionResult(
        prediction=prediction,
        probability_class_0=prob_0,
        probability_class_1=prob_1,
        confidence=max(prob_0, prob_1),
        model_version=bundle.version,
        model_alias=bundle.alias,
        latency_ms=latency_ms,
        features_used=feature_cols,
        warnings=warnings_list,
    )


# ---------------------------------------------------------------------------
# Public: batch prediction
# ---------------------------------------------------------------------------


def predict_batch(records: list[dict]) -> list[PredictionResult]:
    """
    Run predict() on a list of feature dicts.
    Errors on individual records are caught and surfaced as warnings rather
    than aborting the entire batch.
    """
    results = []
    for i, rec in enumerate(records):
        try:
            results.append(predict(rec))
        except Exception as exc:  # noqa: BLE001
            logger.error("predict_batch: record %d failed — %s", i, exc)
            results.append(
                PredictionResult(
                    prediction=-1,
                    probability_class_0=0.0,
                    probability_class_1=0.0,
                    confidence=0.0,
                    model_version=_bundle.version if _bundle else "unknown",
                    model_alias="error",
                    latency_ms=0.0,
                    warnings=[f"Inference error on record {i}: {exc}"],
                )
            )
    return results


# ---------------------------------------------------------------------------
# Public: model info
# ---------------------------------------------------------------------------


def get_model_info() -> dict:
    """
    Return metadata about the currently loaded model bundle.
    Safe to call even if no model is loaded (returns status: unloaded).
    """
    if _bundle is None:
        return {"status": "unloaded", "version": None, "alias": None}

    return {
        "status": "loaded",
        "version": _bundle.version,
        "alias": _bundle.alias,
        "loaded_at": _bundle.loaded_at,
        "registry_mtime": _bundle.registry_mtime,
        "model_type": type(_bundle.model).__name__,
        "pipeline_type": type(_bundle.pipeline).__name__,
        "artifacts_dir": str(ARTIFACTS_DIR),
    }


# ---------------------------------------------------------------------------
# Module self-test (python -m api.predict)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    try:
        logger.info("Pre-load info: %s", get_model_info())
        bundle = _ensure_loaded()
        logger.info("Bundle loaded: version=%s alias=%s", bundle.version, bundle.alias)
        logger.info("reload_if_stale() → %s", reload_if_stale())
        logger.info("Model info: %s", json.dumps(get_model_info(), indent=2, default=str))
        logger.info("predict.py self-test passed.")
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.warning("Artifacts not present yet (expected during dev): %s", exc)
        logger.info("Import structure OK — run training pipeline to generate artifacts.")
        sys.exit(0)