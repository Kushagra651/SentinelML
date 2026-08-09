"""
data/features.py
----------------
Feature engineering pipeline for the UCI Adult Income dataset.

Takes a validated raw DataFrame (output of data/ingest.py) and produces
a model-ready feature matrix.

Steps
-----
1. Type casting      — enforce correct dtypes
2. Log transform     — log1p on right-skewed capital_gain / capital_loss
3. Standard scaling  — StandardScaler on all 6 numeric columns
4. Ordinal encoding  — OrdinalEncoder on all 8 categorical columns
5. Target separation — income column separated before transform (no leakage)

All transforms are stateful (fitted on training data, applied at inference).
Use FeaturePipeline.fit_transform(df) during training and
FeaturePipeline.transform(df) at inference time.

Public API
----------
    FeaturePipeline.fit_transform(df) -> pd.DataFrame
    FeaturePipeline.transform(df)     -> pd.DataFrame
    FeaturePipeline.save(path)        -> None
    FeaturePipeline.load(path)        -> FeaturePipeline
    FeaturePipeline.feature_names()   -> list[str]
    build_features(df, pipeline, fit) -> tuple[pd.DataFrame, pd.Series, FeaturePipeline]
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — UCI Adult Income, 14 features
# ---------------------------------------------------------------------------

# Right-skewed columns — log1p before scaling
LOG_TRANSFORM_COLS: list[str] = [
    "capital_gain",
    "capital_loss",
]

# All 6 numeric columns (post log-transform where applicable)
NUMERIC_COLS: list[str] = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

# All 8 categorical columns
CATEGORICAL_COLS: list[str] = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

TARGET_COL: str = "income"

# Default pipeline save path (inside container: /app/artifacts/)
DEFAULT_PIPELINE_PATH: Path = Path("artifacts/feature_pipeline.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce correct dtypes so downstream transforms never break."""
    df = df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation for numerics, mode for categoricals — per batch."""
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.debug("Imputed '%s' with median=%.4f", col, median_val)

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
            df[col] = df[col].fillna(mode_val)
            logger.debug("Imputed '%s' with mode='%s'", col, mode_val)

    return df


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------


class FeaturePipeline:
    """
    Stateful feature engineering pipeline for UCI Adult Income.

    Usage — training
    ----------------
        pipeline = FeaturePipeline()
        X_train = pipeline.fit_transform(df_train)   # df_train must NOT include 'income'
        pipeline.save()

    Usage — inference
    -----------------
        pipeline = FeaturePipeline.load()
        X = pipeline.transform(df_new)
    """

    def __init__(self) -> None:
        self._scaler: Optional[StandardScaler] = None
        self._encoder: Optional[OrdinalEncoder] = None
        self._numeric_cols_present: list[str] = []
        self._cat_cols_present: list[str] = []
        self._output_columns: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal: shared pre-processing (no fitted state required)
    # ------------------------------------------------------------------

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Steps that don't require fitted state."""
        df = _cast_types(df)
        df = _impute(df)

        # Log1p on skewed columns before scaling
        for col in LOG_TRANSFORM_COLS:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        return df

    # ------------------------------------------------------------------
    # fit_transform — training only
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the pipeline on df and return the transformed feature matrix.
        df must NOT contain the target column 'income' — separate it first
        using build_features() or manually with df.drop(columns=['income']).
        """
        if TARGET_COL in df.columns:
            raise ValueError(
                f"Target column '{TARGET_COL}' found in input to fit_transform(). "
                "Drop it before calling: df.drop(columns=['income'])"
            )

        logger.info("Fitting FeaturePipeline on %d rows.", len(df))
        df = self._base_transform(df)

        # Fit + apply StandardScaler on numeric columns
        self._numeric_cols_present = [c for c in NUMERIC_COLS if c in df.columns]
        self._scaler = StandardScaler()
        df[self._numeric_cols_present] = self._scaler.fit_transform(
            df[self._numeric_cols_present]
        )

        # Fit + apply OrdinalEncoder on categorical columns
        self._cat_cols_present = [c for c in CATEGORICAL_COLS if c in df.columns]
        self._encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df[self._cat_cols_present] = self._encoder.fit_transform(
            df[self._cat_cols_present]
        )

        self._output_columns = self._numeric_cols_present + self._cat_cols_present
        self._is_fitted = True

        logger.info(
            "FeaturePipeline fitted. Output shape: %s. Columns: %s",
            df[self._output_columns].shape,
            self._output_columns,
        )
        return df[self._output_columns]

    # ------------------------------------------------------------------
    # transform — inference / evaluation
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the already-fitted pipeline to df.
        Raises RuntimeError if called before fit_transform.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeaturePipeline has not been fitted. "
                "Call fit_transform() on training data first, or load() a saved pipeline."
            )

        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        df = self._base_transform(df)

        # Apply (don't re-fit) scaler
        num_present = [c for c in self._numeric_cols_present if c in df.columns]
        if num_present:
            df[num_present] = self._scaler.transform(df[num_present])  # type: ignore[union-attr]

        # Apply (don't re-fit) encoder
        cat_present = [c for c in self._cat_cols_present if c in df.columns]
        if cat_present:
            df[cat_present] = self._encoder.transform(df[cat_present])  # type: ignore[union-attr]

        return df[self._output_columns]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str = DEFAULT_PIPELINE_PATH) -> None:
        """Pickle the fitted pipeline to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("FeaturePipeline saved to %s", path)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PIPELINE_PATH) -> "FeaturePipeline":
        """Load a previously saved pipeline from path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No pipeline found at '{path}'.")
        with open(path, "rb") as f:
            pipeline: FeaturePipeline = pickle.load(f)
        logger.info("FeaturePipeline loaded from %s", path)
        return pipeline

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def feature_names(self) -> list[str]:
        """
        Return the list of output feature column names after transform.
        NOTE: call with parentheses — this is a method, not a property.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted yet.")
        return list(self._output_columns)


# ---------------------------------------------------------------------------
# Convenience function — used by training/train.py
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    pipeline: Optional[FeaturePipeline] = None,
    *,
    fit: bool = False,
) -> tuple[pd.DataFrame, pd.Series, FeaturePipeline]:
    """
    High-level helper: separate target, apply pipeline, return (X, y, pipeline).

    Parameters
    ----------
    df : pd.DataFrame
        Validated raw DataFrame including the 'income' target column.
    pipeline : FeaturePipeline, optional
        Existing pipeline; a new one is created if None and fit=True.
    fit : bool
        If True → fit_transform (training). If False → transform (inference).

    Returns
    -------
    X : pd.DataFrame   — transformed feature matrix (14 columns, numeric)
    y : pd.Series      — target labels (">50K" / "<=50K")
    pipeline : FeaturePipeline
    """
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in DataFrame. "
            "Ensure you're passing the full ingested DataFrame."
        )

    if pipeline is None:
        pipeline = FeaturePipeline()

    # Separate target BEFORE transform — prevents data leakage
    y = df[TARGET_COL].copy()
    X_df = df.drop(columns=[TARGET_COL])

    if fit:
        X = pipeline.fit_transform(X_df)
    else:
        X = pipeline.transform(X_df)

    return X, y, pipeline
