"""
data/validate.py
----------------
Validates raw ingested DataFrames before feature engineering.

Checks performed
----------------
* Schema conformance  — all 15 expected columns present (14 features + target)
* Nullability         — no missing values after cleaning
* Value ranges        — numeric columns within UCI-defined bounds
* Categorical levels  — categorical columns contain only known labels
* Duplicate rows      — warns when fully-duplicate rows are detected
* Row-count guard     — raises if the batch is suspiciously small
* Target labels       — income column contains only ">50K" / "<=50K"

All failures are collected and returned as a structured ValidationReport so
callers can decide whether to hard-fail or log-and-continue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema contract — UCI Adult Income, 14 features + 1 target
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",  # target
]

# Numeric bounds — UCI dataset actual value ranges
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (17.0, 90.0),
    "fnlwgt": (10_000.0, 1_500_000.0),
    "education_num": (1.0, 16.0),
    "capital_gain": (0.0, 99_999.0),
    "capital_loss": (0.0, 4_356.0),
    "hours_per_week": (1.0, 99.0),
}

# Allowed categorical levels — derived from UCI dataset
CATEGORICAL_LEVELS: dict[str, set[str]] = {
    "workclass": {
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked",
    },
    "education": {
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
        "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
    },
    "marital_status": {
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse",
    },
    "occupation": {
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces",
    },
    "relationship": {
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried",
    },
    "race": {
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black",
    },
    "sex": {"Male", "Female"},
    "income": {">50K", "<=50K"},
}

MIN_BATCH_ROWS: int = 10


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    column: str
    check: str
    detail: str


@dataclass
class ValidationReport:
    passed: bool = True
    hard_failures: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    invalid_row_count: int = 0

    def add_error(self, column: str, check: str, detail: str) -> None:
        self.hard_failures.append(
            ValidationError(column=column, check=check, detail=detail)
        )
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning("Validation warning: %s", msg)

    @property
    def errors(self) -> list[ValidationError]:
        """Alias for hard_failures — backwards compatibility."""
        return self.hard_failures

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status} | rows={self.row_count} "
            f"| errors={len(self.hard_failures)} | warnings={len(self.warnings)}"
        )


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def _check_row_count(df: pd.DataFrame, report: ValidationReport) -> None:
    """Fail fast if the batch is too small to be meaningful."""
    report.row_count = len(df)
    if len(df) < MIN_BATCH_ROWS:
        report.add_error(
            column="*",
            check="row_count",
            detail=f"Batch has only {len(df)} rows; minimum is {MIN_BATCH_ROWS}.",
        )


def _check_schema(df: pd.DataFrame, report: ValidationReport) -> None:
    """Verify all expected columns are present."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        report.add_error(
            column=str(missing),
            check="schema_columns",
            detail=f"Missing required columns: {missing}",
        )

    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra:
        report.add_warning(f"Extra columns not in schema (will be ignored): {extra}")


def _check_nullability(df: pd.DataFrame, report: ValidationReport) -> None:
    """Ensure no missing values exist after cleaning."""
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            report.add_error(
                column=str(col),
                check="nullability",
                detail=f"Found {count} null values.",
            )


def _check_numeric_bounds(df: pd.DataFrame, report: ValidationReport) -> None:
    """Flag rows where numeric values fall outside expected UCI bounds."""
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out_of_range = series[(series < lo) | (series > hi)]
        if not out_of_range.empty:
            report.invalid_row_count += len(out_of_range)
            report.add_error(
                column=col,
                check="numeric_bounds",
                detail=(
                    f"{len(out_of_range)} values outside [{lo}, {hi}]. "
                    f"Sample indices: {out_of_range.index[:5].tolist()}"
                ),
            )


def _check_categorical_levels(df: pd.DataFrame, report: ValidationReport) -> None:
    """Detect unseen category labels."""
    for col, allowed in CATEGORICAL_LEVELS.items():
        if col not in df.columns:
            continue
        actual = set(df[col].dropna().unique())
        unknown = actual - allowed
        if unknown:
            report.add_warning(
                f"Column '{col}' contains unknown categories: {unknown}."
            )


def _check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    """Warn on fully-duplicate rows (UCI has no primary key column)."""
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        report.add_warning(f"Found {dup_count} fully-duplicate rows.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_dataframe(
    df: pd.DataFrame, *, raise_on_failure: bool = False
) -> ValidationReport:
    """
    Run all validation checks on df and return a ValidationReport.

    Parameters
    ----------
    df : pd.DataFrame
        The ingested DataFrame to validate.
    raise_on_failure : bool
        If True, raises ValueError when any hard check fails.

    Returns
    -------
    ValidationReport
        Structured report with .passed, .hard_failures, and .warnings.
    """
    report = ValidationReport()

    _check_row_count(df, report)
    _check_schema(df, report)
    _check_nullability(df, report)
    _check_numeric_bounds(df, report)
    _check_categorical_levels(df, report)
    _check_duplicates(df, report)

    logger.info(report.summary())

    if not report.passed and raise_on_failure:
        error_details = "\n".join(
            f"  [{e.column}] {e.check}: {e.detail}" for e in report.hard_failures
        )
        raise ValueError(f"Data validation failed:\n{error_details}")

    return report


def validate_single_record(record: dict[str, Any]) -> ValidationReport:
    """
    Validate a single prediction-time feature dict.
    Skips the row-count guard since single records are always below MIN_BATCH_ROWS.
    """
    df = pd.DataFrame([record])

    global MIN_BATCH_ROWS
    original_min = MIN_BATCH_ROWS
    MIN_BATCH_ROWS = 1
    try:
        report = validate_dataframe(df, raise_on_failure=False)
    finally:
        MIN_BATCH_ROWS = original_min

    return report