"""Redis (Feast online) vs Parquet lookup latency + skew check. Logs to MLflow."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

from data.features import FeaturePipeline
from experiments.mlflow_utils import log_run

STORE_DIR = Path("/app/artifacts/feature_store")
SRC = STORE_DIR / "train_feast.parquet"
OUT = Path("experiments/results")
NUM = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
CAT = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
COLS = NUM + CAT
FEATS = [f"adult_raw:{c}" for c in COLS]
SIZES = [1, 10, 100]
rng = np.random.default_rng(42)

store = FeatureStore(repo_path="/app/experiments/feast_repo")
mem = pd.read_parquet(SRC).set_index("person_id")  # cached in memory
N = len(mem)


def ids(k):
    return [int(i) for i in rng.choice(N, size=k, replace=False)]


def redis_lookup(k):
    return store.get_online_features(
        features=FEATS, entity_rows=[{"person_id": i} for i in ids(k)]
    ).to_dict()


def parquet_per_request(k):
    df = pd.read_parquet(SRC)  # full read every request
    return df[df["person_id"].isin(ids(k))]


def memory_lookup(k):
    return mem.loc[ids(k), COLS]


def bench(fn, k, n):
    for _ in range(3):
        fn(k)  # warm-up
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn(k)
        t.append((time.perf_counter() - s) * 1000)
    a = np.array(t)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "n": n,
    }


results, metrics = {}, {}
for name, fn, n in [
    ("redis_feast", redis_lookup, 200),
    ("parquet_per_request", parquet_per_request, 30),
    ("memory_dataframe", memory_lookup, 200),
]:
    for k in SIZES:
        r = bench(fn, k, n)
        results[f"{name}_n{k}"] = r
        for p in ("p50", "p95", "p99"):
            metrics[f"{name}_n{k}_{p}_ms"] = r[p]
        print(
            f"{name:20s} entities={k:3d} p50={r['p50']:.2f} p95={r['p95']:.2f} p99={r['p99']:.2f} ms"
        )

# ---- skew check: online vs parquet, raw then after FeaturePipeline ----
sample = sorted(ids(1000))
online = pd.DataFrame(
    store.get_online_features(
        features=FEATS, entity_rows=[{"person_id": i} for i in sample]
    ).to_dict()
)
online = online.set_index("person_id").loc[sample, COLS]
offline = mem.loc[sample, COLS]
raw_equal = bool(online.astype(str).equals(offline.astype(str)))

reg = json.loads(Path("/app/artifacts/production_model.json").read_text())
pipe = FeaturePipeline.load(reg["pipeline_path"])
xa, xb = pipe.transform(online.copy()), pipe.transform(offline.copy())
pipe_equal = bool(np.allclose(xa.values, xb.values))
print(
    f"skew: raw_equal={raw_equal} pipeline_output_equal={pipe_equal} (n={len(sample)})"
)

metrics["skew_raw_equal"] = int(raw_equal)
metrics["skew_pipeline_equal"] = int(pipe_equal)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "feature_store_bench.json").write_text(
    json.dumps(
        {"latency": results, "raw_equal": raw_equal, "pipeline_equal": pipe_equal},
        indent=2,
    )
)
rid = log_run(
    params={"n_rows": N, "sizes": SIZES},
    metrics=metrics,
    artifacts=[OUT / "feature_store_bench.json"],
    tags={"task": "feature_store"},
    run_name="feature_store_bench",
    experiment_name="feature_store",
)
print("mlflow run:", rid)
