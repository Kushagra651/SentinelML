"""Adds synthetic person_id + event_timestamp to train.parquet (Feast offline source)."""

import os
from pathlib import Path

import pandas as pd

STORE = Path(os.getenv("FEATURE_STORE_DIR", "data/feature_store"))
NUM = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

df = pd.read_parquet(STORE / "train.parquet").drop(columns=["income"])
df[NUM] = df[NUM].astype("int64")
df.insert(0, "person_id", range(len(df)))  # synthetic: row number
df["person_id"] = df["person_id"].astype("int64")
df["event_timestamp"] = pd.Timestamp("2026-11-01", tz="UTC")  # synthetic, fixed
out = STORE / "train_feast.parquet"
df.to_parquet(out, index=False)
print(f"wrote {out}: {df.shape}")
