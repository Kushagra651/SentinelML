"""
data/drift_injector.py
----------------------
Simulates realistic data drift scenarios for testing the monitoring pipeline.

Drift types supported
---------------------
1. Covariate drift     — input feature distribution shifts (gradual / sudden)
2. Label drift         — target class distribution shifts
3. Concept drift       — relationship between features and label changes
4. Missing value drift — sudden spike in nulls for specific columns
5. Schema drift        — unexpected new / dropped / renamed columns
6. Categorical drift   — new unseen category labels appear
7. Temporal drift      — no-op for UCI Adult (no timestamp column); kept for API compatibility

Each injector is a pure function:
    inject_*(df, **params) -> pd.DataFrame

A high-level inject_drift dispatcher accepts a DriftConfig dataclass
so Airflow DAGs and tests can drive injection declaratively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------


class DriftType(str, Enum):
    COVARIATE = "covariate"
    LABEL = "label"
    CONCEPT = "concept"
    MISSING_VALUE = "missing_value"
    SCHEMA = "schema"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    NONE = "none"


@dataclass
class DriftConfig:
    """
    Declarative configuration for a single drift injection run.

    Parameters
    ----------
    drift_type:
        Which drift scenario to apply.
    intensity:
        0.0 = no drift, 1.0 = maximum drift. Interpreted per injector.
    affected_columns:
        Columns to target. Empty list = injector picks sensible defaults.
    gradual:
        If True, drift is applied progressively across rows.
    seed:
        Random seed for reproducibility.
    extra:
        Injector-specific kwargs passed through as-is.
    """

    drift_type: DriftType = DriftType.NONE
    intensity: float = 0.3
    affected_columns: list[str] = field(default_factory=list)
    gradual: bool = False
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default column sets — UCI Adult Income dataset
# ---------------------------------------------------------------------------

_DEFAULT_NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

_DEFAULT_CAT_COLS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

_LABEL_COL = "income"  # UCI Adult target column: ">50K" / "<=50K"
_TIMESTAMP_COL = "timestamp"  # UCI Adult has no timestamp — kept for API compatibility


# ---------------------------------------------------------------------------
# 1. Covariate drift
# ---------------------------------------------------------------------------


def inject_covariate_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    affected_columns: Optional[list[str]] = None,
    gradual: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Shift numeric feature distributions by adding scaled noise / bias.

    intensity controls the magnitude of the shift as a fraction of each
    column's standard deviation (e.g. 0.3 → shift mean by 0.3 * std).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    cols = affected_columns or [c for c in _DEFAULT_NUMERIC_COLS if c in df.columns]
    n = len(df)

    for col in cols:
        if col not in df.columns:
            logger.warning("Covariate drift: column '%s' not found, skipping.", col)
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        std = series.std(skipna=True)
        if pd.isna(std) or std == 0:
            std = 1.0

        shift = intensity * std

        if gradual:
            progressive_shift = np.linspace(0, shift, n)
            noise = rng.normal(loc=0, scale=std * 0.05, size=n)
            df[col] = series + progressive_shift + noise
        else:
            n_affected = max(1, int(n * intensity))
            idx = rng.choice(n, size=n_affected, replace=False)
            noise = rng.normal(loc=shift, scale=std * 0.1, size=n_affected)
            df.loc[df.index[idx], col] = series.iloc[idx].values + noise

        logger.debug(
            "Covariate drift injected into '%s' | shift=%.4f | gradual=%s",
            col,
            shift,
            gradual,
        )

    return df


# ---------------------------------------------------------------------------
# 2. Label drift
# ---------------------------------------------------------------------------


def inject_label_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    target_positive_rate: Optional[float] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Flip a fraction of income labels to simulate class distribution shift.

    For UCI Adult: positive class is ">50K", negative is "<=50K".
    target_positive_rate overrides intensity when provided.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    if _LABEL_COL not in df.columns:
        logger.warning("Label drift: '%s' column not found. Skipping.", _LABEL_COL)
        return df

    labels = df[_LABEL_COL].copy()
    n = len(labels)
    pos_label, neg_label = ">50K", "<=50K"

    if target_positive_rate is not None:
        current_pos = (labels == pos_label).sum()
        desired_pos = int(target_positive_rate * n)
        delta = desired_pos - current_pos

        if delta > 0:
            neg_idx = labels[labels == neg_label].index.tolist()
            flip_idx = rng.choice(neg_idx, size=min(delta, len(neg_idx)), replace=False)
            labels.loc[flip_idx] = pos_label
        elif delta < 0:
            pos_idx = labels[labels == pos_label].index.tolist()
            flip_idx = rng.choice(
                pos_idx, size=min(-delta, len(pos_idx)), replace=False
            )
            labels.loc[flip_idx] = neg_label
    else:
        n_flip = max(1, int(n * intensity))
        flip_idx = rng.choice(n, size=n_flip, replace=False)
        labels.iloc[flip_idx] = labels.iloc[flip_idx].map(
            {pos_label: neg_label, neg_label: pos_label}
        )

    df[_LABEL_COL] = labels
    new_rate = (df[_LABEL_COL] == pos_label).mean()
    logger.debug("Label drift injected. New positive rate=%.4f", new_rate)
    return df


# ---------------------------------------------------------------------------
# 3. Concept drift
# ---------------------------------------------------------------------------


def inject_concept_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    gradual: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Break the feature→label relationship by re-assigning income labels for a
    subset of rows based on a different decision boundary.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    if _LABEL_COL not in df.columns:
        logger.warning("Concept drift: '%s' not found. Skipping.", _LABEL_COL)
        return df

    n = len(df)
    pos_label, neg_label = ">50K", "<=50K"

    if gradual:
        flip_prob = np.linspace(0, intensity, n)
        flip_mask = rng.random(n) < flip_prob
    else:
        n_affected = max(1, int(n * intensity))
        flip_mask = np.zeros(n, dtype=bool)
        flip_mask[rng.choice(n, size=n_affected, replace=False)] = True

    df.loc[flip_mask, _LABEL_COL] = df.loc[flip_mask, _LABEL_COL].map(
        {pos_label: neg_label, neg_label: pos_label}
    )

    # Add mild covariate noise to the same rows
    for col in [c for c in _DEFAULT_NUMERIC_COLS if c in df.columns]:
        series = pd.to_numeric(df[col], errors="coerce")
        std = series.std(skipna=True) or 1.0
        noise = rng.normal(0, std * 0.2 * intensity, size=int(flip_mask.sum()))
        df.loc[flip_mask, col] = series[flip_mask].values + noise

    logger.debug("Concept drift injected. Rows affected: %d / %d", flip_mask.sum(), n)
    return df


# ---------------------------------------------------------------------------
# 4. Missing value drift
# ---------------------------------------------------------------------------


def inject_missing_value_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    affected_columns: Optional[list[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Introduce NaN values into specified columns.
    intensity is the fraction of rows that will have nulls introduced.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    cols = affected_columns or [c for c in _DEFAULT_NUMERIC_COLS if c in df.columns]
    n = len(df)

    for col in cols:
        if col not in df.columns:
            continue
        n_null = max(1, int(n * intensity))
        null_idx = rng.choice(n, size=n_null, replace=False)
        df.iloc[null_idx, df.columns.get_loc(col)] = np.nan
        logger.debug("Missing value drift: %d nulls injected into '%s'.", n_null, col)

    return df


# ---------------------------------------------------------------------------
# 5. Schema drift
# ---------------------------------------------------------------------------


def inject_schema_drift(
    df: pd.DataFrame,
    *,
    drop_columns: Optional[list[str]] = None,
    add_columns: Optional[dict[str, Any]] = None,
    rename_columns: Optional[dict[str, str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate schema changes: drop columns, add new ones, rename existing ones.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    if drop_columns:
        existing = [c for c in drop_columns if c in df.columns]
        df = df.drop(columns=existing)
        logger.debug("Schema drift: dropped columns %s.", existing)

    if add_columns:
        for col, fill in add_columns.items():
            df[col] = fill(len(df), rng) if callable(fill) else fill
            logger.debug("Schema drift: added column '%s'.", col)

    if rename_columns:
        existing_renames = {k: v for k, v in rename_columns.items() if k in df.columns}
        df = df.rename(columns=existing_renames)
        logger.debug("Schema drift: renamed columns %s.", existing_renames)

    return df


# ---------------------------------------------------------------------------
# 6. Categorical drift
# ---------------------------------------------------------------------------


def inject_categorical_drift(
    df: pd.DataFrame,
    *,
    intensity: float = 0.3,
    affected_columns: Optional[list[str]] = None,
    new_categories: Optional[dict[str, list[str]]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Introduce unseen category labels into categorical columns.
    Defaults to generic unknown_<col>_<i> labels if new_categories not provided.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    cols = affected_columns or [c for c in _DEFAULT_CAT_COLS if c in df.columns]
    n = len(df)

    for col in cols:
        if col not in df.columns:
            continue
        new_labels = (new_categories or {}).get(
            col, [f"unknown_{col}_0", f"unknown_{col}_1"]
        )
        n_affected = max(1, int(n * intensity))
        idx = rng.choice(n, size=n_affected, replace=False)
        chosen_labels = rng.choice(new_labels, size=n_affected)
        df.iloc[idx, df.columns.get_loc(col)] = chosen_labels
        logger.debug(
            "Categorical drift: %d rows in '%s' set to unseen labels %s.",
            n_affected,
            col,
            new_labels,
        )

    return df


# ---------------------------------------------------------------------------
# 7. Temporal drift — no-op for UCI Adult (no timestamp column)
# ---------------------------------------------------------------------------


def inject_temporal_drift(
    df: pd.DataFrame,
    *,
    gap_hours: float = 72.0,
    out_of_order_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Introduce temporal anomalies. UCI Adult Income has no timestamp column,
    so this is a no-op that logs a warning and returns the DataFrame unchanged.
    Kept for API compatibility with the DriftConfig dispatcher.
    """
    if _TIMESTAMP_COL not in df.columns:
        logger.warning(
            "Temporal drift: '%s' column not found in DataFrame (expected for UCI Adult). "
            "Returning DataFrame unchanged.",
            _TIMESTAMP_COL,
        )
        return df.copy()

    rng = np.random.default_rng(seed)
    df = df.copy()
    ts = pd.to_datetime(df[_TIMESTAMP_COL], utc=True, errors="coerce")
    n = len(ts)
    midpoint = n // 2

    gap = pd.Timedelta(hours=gap_hours)
    ts.iloc[midpoint:] = ts.iloc[midpoint:] + gap

    n_shuffle = max(1, int(n * out_of_order_fraction))
    idx = rng.choice(n, size=n_shuffle, replace=False)
    shuffled_vals = ts.iloc[idx].values.copy()
    rng.shuffle(shuffled_vals)
    ts.iloc[idx] = shuffled_vals

    df[_TIMESTAMP_COL] = ts
    return df


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def inject_drift(df: pd.DataFrame, config: DriftConfig) -> pd.DataFrame:
    """
    Central dispatcher — routes to the correct injector based on config.drift_type.
    """
    logger.info(
        "Injecting drift | type=%s | intensity=%.2f | gradual=%s | seed=%d",
        config.drift_type,
        config.intensity,
        config.gradual,
        config.seed,
    )

    kwargs: dict[str, Any] = dict(
        intensity=config.intensity,
        seed=config.seed,
        **config.extra,
    )

    match config.drift_type:
        case DriftType.NONE:
            return df.copy()

        case DriftType.COVARIATE:
            return inject_covariate_drift(
                df,
                affected_columns=config.affected_columns or None,
                gradual=config.gradual,
                **kwargs,
            )

        case DriftType.LABEL:
            return inject_label_drift(df, **kwargs)

        case DriftType.CONCEPT:
            return inject_concept_drift(df, gradual=config.gradual, **kwargs)

        case DriftType.MISSING_VALUE:
            return inject_missing_value_drift(
                df,
                affected_columns=config.affected_columns or None,
                **kwargs,
            )

        case DriftType.SCHEMA:
            return inject_schema_drift(
                df,
                drop_columns=config.extra.get("drop_columns"),
                add_columns=config.extra.get("add_columns"),
                rename_columns=config.extra.get("rename_columns"),
                seed=config.seed,
            )

        case DriftType.CATEGORICAL:
            return inject_categorical_drift(
                df,
                affected_columns=config.affected_columns or None,
                new_categories=config.extra.get("new_categories"),
                **{k: v for k, v in kwargs.items() if k != "new_categories"},
            )

        case DriftType.TEMPORAL:
            return inject_temporal_drift(
                df,
                gap_hours=config.extra.get("gap_hours", 72.0),
                out_of_order_fraction=config.extra.get("out_of_order_fraction", 0.1),
                seed=config.seed,
            )

        case _:
            raise ValueError(f"Unknown drift type: '{config.drift_type}'")


# ---------------------------------------------------------------------------
# Composite injector
# ---------------------------------------------------------------------------


def inject_composite_drift(
    df: pd.DataFrame,
    configs: list[DriftConfig],
) -> pd.DataFrame:
    """
    Apply multiple drift injections sequentially (left-to-right).
    Used by Airflow DAGs to chain multiple drift types in one run.
    """
    for cfg in configs:
        df = inject_drift(df, cfg)
    return df
