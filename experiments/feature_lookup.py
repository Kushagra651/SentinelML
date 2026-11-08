"""Feature lookup: Redis (Feast) first, Parquet fallback. Not used by /predict."""

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
STORE_DIR = Path(os.getenv("FEATURE_STORE_DIR", "data/feature_store"))
COLS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def _from_parquet(ids: list[int]) -> pd.DataFrame:
    df = pd.read_parquet(STORE_DIR / "train_feast.parquet")
    return df[df["person_id"].isin(ids)][["person_id"] + COLS].reset_index(drop=True)


def _from_redis(ids: list[int]) -> pd.DataFrame:
    from feast import FeatureStore  # noqa: PLC0415

    store = FeatureStore(
        repo_path=os.getenv("FEAST_REPO", "/app/experiments/feast_repo")
    )
    out = store.get_online_features(
        features=[f"adult_raw:{c}" for c in COLS],
        entity_rows=[{"person_id": i} for i in ids],
    ).to_dict()
    return pd.DataFrame(out)


def get_features(ids: list[int]) -> pd.DataFrame:
    try:
        df = _from_redis(ids)
        if df[COLS].isna().all(axis=1).any():
            raise RuntimeError("keys missing in online store")
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("Redis lookup failed (%s) - falling back to Parquet.", exc)
        return _from_parquet(ids)
