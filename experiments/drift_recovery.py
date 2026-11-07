"""
experiments/drift_recovery.py
-----------------------------
TASK 1: Drift -> Degradation -> Recovery.

For 3 covariate-drift severities (mild / moderate / severe) it measures
accuracy, ROC-AUC and PR-AUC for:
  A. old model on CLEAN test data            (baseline)
  B. old model on DRIFTED test data          (degradation)
  C. retrained model on DRIFTED test data    (recovery)
  D. retrained model on CLEAN test data      (cost of adapting)

Setup (mirrors training/train.py exactly):
  - features fit on full train split, 80/20 stratified split, seed 42
  - GradientBoostingClassifier with DEFAULT_PARAMS from training.train
Drift is injected with data/drift_injector.py (DriftType.COVARIATE).
Labels are NOT touched, so this is pure covariate drift.
Test drift and train drift use different seeds. Repeated --repeats times.

Touches NO existing file, NO registry, NO database.

Run from repo root:
    python -m experiments.drift_recovery --repeats 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from data.drift_injector import DriftConfig, DriftType, inject_drift  # noqa: E402
from data.features import NUMERIC_COLS, build_features  # noqa: E402
from data.ingest import load_from_feature_store  # noqa: E402
from training.train import DEFAULT_PARAMS, RANDOM_SEED, VALIDATION_SPLIT  # noqa: E402
from experiments.mlflow_utils import log_run  # noqa: E402

logging.getLogger().setLevel(logging.WARNING)
for _name in ("data.ingest", "data.features", "training.train", "root"):
    logging.getLogger(_name).setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

RESULTS_DIR = Path("experiments/results")
SEVERITIES = {"mild": 0.1, "moderate": 0.3, "severe": 0.6}  # injector `intensity`
METRICS = ["accuracy", "roc_auc", "pr_auc"]


# ----------------------------------------------------------------------------
def to_binary(y: pd.Series) -> np.ndarray:
    return (y == ">50K").astype(int).values


def make_drift(df: pd.DataFrame, intensity: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float)  # avoid int<-float assign errors
    cfg = DriftConfig(
        drift_type=DriftType.COVARIATE, intensity=intensity, gradual=False, seed=seed
    )
    return inject_drift(df, cfg)


def fit_model(train_df: pd.DataFrame):
    """Same recipe as training.train.run_training (minus I/O)."""
    X, y, pipeline = build_features(train_df, fit=True)
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    model = GradientBoostingClassifier(**DEFAULT_PARAMS)
    model.fit(X_tr, to_binary(y_tr))
    return model, pipeline


def score(model, pipeline, df: pd.DataFrame) -> dict:
    X, y, _ = build_features(df, pipeline=pipeline, fit=False)
    prob = model.predict_proba(X)[:, 1]
    yb = to_binary(y)
    return {
        "accuracy": float(accuracy_score(yb, prob >= 0.5)),
        "roc_auc": float(roc_auc_score(yb, prob)),
        "pr_auc": float(average_precision_score(yb, prob)),
    }


# ----------------------------------------------------------------------------
def run(repeats: int) -> pd.DataFrame:
    train_df = load_from_feature_store("train")
    test_df = load_from_feature_store("test")
    print(
        f"train={len(train_df)} test={len(test_df)} "
        f"test_pos_rate={(test_df['income'] == '>50K').mean():.4f}"
    )

    t0 = time.perf_counter()
    base_model, base_pipe = fit_model(train_df)
    print(f"base model trained in {time.perf_counter() - t0:.1f}s")
    base_clean = score(base_model, base_pipe, test_df)
    print("A. old model, clean test:", {k: round(v, 4) for k, v in base_clean.items()})

    rows = []
    for sev, intensity in SEVERITIES.items():
        for r in range(repeats):
            t1 = time.perf_counter()
            test_d = make_drift(test_df, intensity, seed=1000 + r)
            train_d = make_drift(train_df, intensity, seed=2000 + r)

            old_drift = score(base_model, base_pipe, test_d)
            new_model, new_pipe = fit_model(train_d)
            new_drift = score(new_model, new_pipe, test_d)
            new_clean = score(new_model, new_pipe, test_df)

            for stage, res in [
                ("A_old_clean", base_clean),
                ("B_old_drifted", old_drift),
                ("C_retrained_drifted", new_drift),
                ("D_retrained_clean", new_clean),
            ]:
                rows.append(
                    {
                        "severity": sev,
                        "intensity": intensity,
                        "repeat": r,
                        "stage": stage,
                        **res,
                    }
                )
            print(
                f"[{sev} rep{r}] old_drift acc={old_drift['accuracy']:.4f} -> "
                f"retrained acc={new_drift['accuracy']:.4f} "
                f"({time.perf_counter() - t1:.0f}s)"
            )
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    g = raw.groupby(["intensity", "severity", "stage"])[METRICS]
    mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
    out = mean.copy()
    for m in METRICS:
        out[f"{m}_std"] = std[m]
    return out.reset_index().sort_values(["intensity", "stage"])


def write_table(summary: pd.DataFrame, repeats: int, path: Path) -> str:
    lines = [
        f"Drift -> Degradation -> Recovery (mean +- std over {repeats} drift seeds)",
        "",
        "| severity | stage | accuracy | ROC-AUC | PR-AUC |",
        "|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['severity']} | {r['stage']} | "
            f"{r['accuracy']:.4f} +- {r['accuracy_std']:.4f} | "
            f"{r['roc_auc']:.4f} +- {r['roc_auc_std']:.4f} | "
            f"{r['pr_auc']:.4f} +- {r['pr_auc_std']:.4f} |"
        )
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def plot(summary: pd.DataFrame, path: Path) -> None:
    stages = ["A_old_clean", "B_old_drifted", "C_retrained_drifted"]
    labels = ["Old model / clean", "Old model / drifted", "Retrained / drifted"]
    colors = ["#4c78a8", "#e45756", "#54a24b"]
    sevs = list(SEVERITIES)
    x = np.arange(len(sevs))
    w = 0.26

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m in zip(axes, METRICS):
        for i, (st, lab, c) in enumerate(zip(stages, labels, colors)):
            sub = summary[summary["stage"] == st].set_index("severity").loc[sevs]
            ax.bar(
                x + (i - 1) * w,
                sub[m],
                w,
                yerr=sub[f"{m}_std"],
                label=lab,
                color=c,
                capsize=3,
            )
        lo = summary[m].min()
        ax.set_ylim(max(0, lo - 0.05), min(1.0, summary[m].max() + 0.03))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n(intensity {SEVERITIES[s]})" for s in sevs])
        ax.set_title(m.replace("_", "-").upper())
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "SentinelML: covariate drift degrades the old model; retraining recovers"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3, help="drift seeds per severity")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = run(args.repeats)
    summary = summarize(raw)

    raw.to_csv(RESULTS_DIR / "drift_recovery_raw.csv", index=False)
    summary.to_csv(RESULTS_DIR / "drift_recovery_summary.csv", index=False)
    text = write_table(summary, args.repeats, RESULTS_DIR / "drift_recovery_table.md")
    plot(summary, RESULTS_DIR / "drift_recovery.png")
    (RESULTS_DIR / "drift_recovery_meta.json").write_text(
        json.dumps(
            {
                "severities": SEVERITIES,
                "repeats": args.repeats,
                "gbm_params": DEFAULT_PARAMS,
                "split": VALIDATION_SPLIT,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        metrics = {}
        for _, r in summary.iterrows():
            for m in METRICS:
                metrics[f"{r['severity']}_{r['stage']}_{m}"] = r[m]
        log_run(
            experiment_name="sentinelml-task1-drift-recovery",
            run_name=f"drift_recovery_repeats{args.repeats}",
            params={
                "repeats": args.repeats,
                "severities": json.dumps(SEVERITIES),
                "split": VALIDATION_SPLIT,
                "random_seed": RANDOM_SEED,
                **{f"gbm_{k}": v for k, v in DEFAULT_PARAMS.items()},
            },
            metrics=metrics,
            artifacts=[
                RESULTS_DIR / "drift_recovery.png",
                RESULTS_DIR / "drift_recovery_table.md",
                RESULTS_DIR / "drift_recovery_meta.json",
            ],
            tags={"task": "1"},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MLflow logging skipped: {exc}")
    print("\n" + text)
    print(f"\nSaved to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
