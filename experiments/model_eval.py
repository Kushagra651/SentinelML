"""
experiments/model_eval.py
-------------------------
TASK 3: model evaluation rigor for the SentinelML GradientBoostingClassifier.

Sections
--------
1. Bootstrap 95% CIs (1000 resamples, percentile) for accuracy, F1, ROC-AUC,
   PR-AUC, Brier on the TEST split. Resamples the test set only (model fixed),
   so the CI captures test-sampling noise, NOT training randomness.
2. Class balance + baselines: majority/prior, logistic regression on the same
   ordinal features, logistic regression with one-hot (fair baseline).
   Paired bootstrap of GBM minus each baseline.
3. Calibration: Brier, Brier skill vs prior, weighted ECE, reliability curve.
4. Per-group metrics for sex and race (n, selection rate, accuracy + CI, TPR,
   FPR, ROC-AUC). NOTE: sex and race are INPUT features of this model.
5. Promotion gate check against real noise:
     - absolute gates (accuracy 0.75, F1 0.70, ROC-AUC 0.75) vs baselines/CI
     - delta gate (0.001 ROC-AUC) vs seed-to-seed noise: train K models that
       differ only by random_state and count how often a statistically
       equivalent model would be promoted (false promotion rate), under the
       current gate and under a paired-bootstrap gate.
6. Test-size justification: CI width vs test size.

The model is trained here with the same recipe as training/train.py
(pipeline fit on full train, 80/20 stratified split, same GBM params,
seed 42), so numbers are for the same recipe, not necessarily the exact
production pickle. Use --use_production to evaluate the pickled production
model instead (needs artifacts/production_model.json readable on the host).

Run:
    python -m experiments.model_eval --quick
    python -m experiments.model_eval
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.features import CATEGORICAL_COLS, NUMERIC_COLS, build_features
from data.ingest import load_from_feature_store

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TARGET = "income"
POS = ">50K"
SEED = 42

# Must match training/train.py DEFAULT_PARAMS (env overrides respected).
GBM_PARAMS = {
    "n_estimators": int(os.getenv("GBM_N_ESTIMATORS", "200")),
    "max_depth": int(os.getenv("GBM_MAX_DEPTH", "4")),
    "learning_rate": float(os.getenv("GBM_LEARNING_RATE", "0.05")),
    "subsample": float(os.getenv("GBM_SUBSAMPLE", "0.8")),
    "random_state": SEED,
}
# Gates copied from training/evaluate.py and training/register_model.py
GATE_ACC, GATE_F1, GATE_AUC = 0.75, 0.70, 0.75
GATE_DELTA = float(os.getenv("MIN_IMPROVEMENT_DELTA", "0.001"))

METRICS = ["accuracy", "f1", "roc_auc", "pr_auc", "brier"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def md(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, row in df.iterrows():
        cells = [
            fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def point_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return {
        "accuracy": float((pred == y).mean()),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
    }


def pct(a: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def boot_metrics(y: np.ndarray, probs: dict, n_boot: int, seed: int) -> dict:
    """Same resample indices for every model -> paired comparisons are valid."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = {k: {m: np.empty(n_boot) for m in METRICS} for k in probs}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for k, p in probs.items():
            pm = point_metrics(yb, p[idx])
            for m in METRICS:
                out[k][m][b] = pm[m]
        if (b + 1) % 200 == 0:
            print(f"    bootstrap {b + 1}/{n_boot}")
    return out


# ---------------------------------------------------------------------------
# data + models
# ---------------------------------------------------------------------------


def make_lr_onehot() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                NUMERIC_COLS,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000))])


def prep_raw_for_lr(df: pd.DataFrame) -> pd.DataFrame:
    d = df.drop(columns=[TARGET], errors="ignore").copy()
    for c in NUMERIC_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("capital_gain", "capital_loss"):
        d[c] = np.log1p(d[c].clip(lower=0))
    for c in CATEGORICAL_COLS:
        d[c] = d[c].astype(str)
    return d


def load_everything(use_production: bool):
    raw_train = load_from_feature_store("train").reset_index(drop=True)
    raw_test = load_from_feature_store("test").reset_index(drop=True)

    X_all, y_all, pipe = build_features(raw_train, fit=True)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
    )
    X_te, y_te_raw, _ = build_features(raw_test, fit=False, pipeline=pipe)
    y_te = (y_te_raw == POS).astype(int).values

    if use_production:
        art = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
        info = json.loads((art / "production_model.json").read_text())
        tag = info["version_tag"]
        mdir = Path(os.getenv("MODEL_DIR", str(art / "models")))
        with open(mdir / f"model_v{tag}.pkl", "rb") as f:
            model = pickle.load(f)
        with open(mdir / f"pipeline_v{tag}.pkl", "rb") as f:
            prod_pipe = pickle.load(f)
        X_te, _, _ = build_features(raw_test, fit=False, pipeline=prod_pipe)
        print(f"using PRODUCTION model v{tag}")
    else:
        t0 = time.perf_counter()
        model = GradientBoostingClassifier(**GBM_PARAMS).fit(X_tr, y_tr)
        print(f"trained GBM in {time.perf_counter() - t0:.1f}s")

    assert list(model.classes_) == ["<=50K", POS], model.classes_
    return raw_train, raw_test, X_tr, y_tr, X_va, y_va, X_te, y_te, model


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def group_table(name, vals, y, p, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for g in sorted(pd.unique(vals)):
        m = vals == g
        yg, pg = y[m], p[m]
        n = int(m.sum())
        pred = (pg >= 0.5).astype(int)
        pos = int((yg == 1).sum())
        neg = n - pos
        accs = []
        if n >= 2:
            for _ in range(n_boot):
                i = rng.integers(0, n, n)
                accs.append(float((pred[i] == yg[i]).mean()))
        lo, hi = pct(np.array(accs)) if accs else (np.nan, np.nan)
        both = pos > 0 and neg > 0
        rows.append(
            {
                "attr": name,
                "group": g,
                "n": n,
                "pos_rate": pos / n if n else np.nan,
                "selection_rate": float(pred.mean()) if n else np.nan,
                "accuracy": float((pred == yg).mean()) if n else np.nan,
                "acc_ci_lo": lo,
                "acc_ci_hi": hi,
                "tpr": float(((pred == 1) & (yg == 1)).sum() / pos) if pos else np.nan,
                "fpr": float(((pred == 1) & (yg == 0)).sum() / neg) if neg else np.nan,
                "roc_auc": float(roc_auc_score(yg, pg)) if both else np.nan,
                "small_n_flag": "n<100" if n < 100 else "",
            }
        )
    return pd.DataFrame(rows)


def ci_width_vs_n(y, p, sizes, n_boot, seed, reps=3):
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        wa, wr = [], []
        for _ in range(reps):
            sub = rng.choice(len(y), size=n, replace=False)
            ys, ps = y[sub], p[sub]
            accs, aucs = [], []
            for _ in range(n_boot):
                i = rng.integers(0, n, n)
                accs.append(float(((ps[i] >= 0.5) == ys[i]).mean()))
                aucs.append(float(roc_auc_score(ys[i], ps[i])))
            lo, hi = pct(np.array(accs))
            wa.append(hi - lo)
            lo, hi = pct(np.array(aucs))
            wr.append(hi - lo)
        acc = float(((p >= 0.5) == y).mean())
        rows.append(
            {
                "n_test": n,
                "acc_ci_width": float(np.mean(wa)),
                "auc_ci_width": float(np.mean(wr)),
                "acc_width_analytic": float(2 * 1.96 * np.sqrt(acc * (1 - acc) / n)),
            }
        )
        print(f"    n={n}: acc width={np.mean(wa):.4f}  auc width={np.mean(wr):.4f}")
    return pd.DataFrame(rows)


def seed_gate_analysis(X_tr, y_tr, X_te, y_te, k_models, n_boot, seed):
    """Train K models that differ ONLY by random_state; test the delta gate."""
    probs = []
    for s in range(SEED, SEED + k_models):
        t0 = time.perf_counter()
        m = GradientBoostingClassifier(**{**GBM_PARAMS, "random_state": s}).fit(
            X_tr, y_tr
        )
        probs.append(m.predict_proba(X_te)[:, 1])
        print(f"    seed {s} trained ({time.perf_counter() - t0:.1f}s)")
    aucs = np.array([roc_auc_score(y_te, p) for p in probs])

    rng = np.random.default_rng(seed)
    n = len(y_te)
    A = np.empty((n_boot, k_models))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for j, p in enumerate(probs):
            A[b, j] = roc_auc_score(y_te[idx], p[idx])

    cur = boot = both = total = 0
    deltas, boot_sd = [], []
    for i in range(k_models):  # incumbent i
        for j in range(k_models):  # candidate j
            if i == j:
                continue
            d_point = aucs[j] - aucs[i]
            d_boot = A[:, j] - A[:, i]
            lo = float(np.percentile(d_boot, 2.5))
            total += 1
            deltas.append(d_point)
            boot_sd.append(float(d_boot.std()))
            c = d_point >= GATE_DELTA
            bgate = lo > 0
            cur += int(c)
            boot += int(bgate)
            both += int(bgate and d_point >= GATE_DELTA)
    return {
        "k_models": k_models,
        "seed_aucs": aucs.tolist(),
        "seed_auc_sd": float(aucs.std(ddof=1)),
        "seed_auc_range": float(aucs.max() - aucs.min()),
        "pair_delta_sd": float(np.std(deltas, ddof=1)),
        "paired_test_noise_sd": float(np.mean(boot_sd)),
        "ordered_pairs": total,
        "false_promo_current_gate": cur / total,
        "false_promo_paired_bootstrap_gate": boot / total,
        "false_promo_bootstrap_plus_margin": both / total,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--k_models", type=int, default=8)
    ap.add_argument("--n_boot_pairs", type=int, default=200)
    ap.add_argument("--n_boot_size", type=int, default=300)
    ap.add_argument("--use_production", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_boot, args.k_models, args.n_boot_pairs, args.n_boot_size = 200, 4, 60, 60

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test, X_tr, y_tr, X_va, y_va, X_te, y_te, model = load_everything(
        args.use_production
    )
    N = len(y_te)
    p_gbm = model.predict_proba(X_te)[:, 1]

    # ---- baselines
    prior = float((y_tr == POS).mean())
    p_major = np.full(N, prior)
    lr_ord = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    p_lr_ord = lr_ord.predict_proba(X_te)[:, 1]
    lr_oh = make_lr_onehot().fit(prep_raw_for_lr(raw_train.loc[X_tr.index]), y_tr)
    p_lr_oh = lr_oh.predict_proba(prep_raw_for_lr(raw_test))[:, 1]

    probs = {
        "gbm": p_gbm,
        "lr_onehot": p_lr_oh,
        "lr_ordinal": p_lr_ord,
        "majority_prior": p_major,
    }

    out: list[str] = ["# Task 3: model evaluation rigor\n"]

    # ---- 2. class balance
    out.append("## Class balance")
    bal = pd.DataFrame(
        [
            {
                "split": "train",
                "n": len(raw_train),
                "pos_rate": float((raw_train[TARGET] == POS).mean()),
            },
            {"split": "test", "n": N, "pos_rate": float(y_te.mean())},
        ]
    )
    out.append(md(bal))
    maj_acc = 1 - float(y_te.mean())
    out.append(f"Majority-class accuracy on test = {maj_acc:.4f}\n")

    # ---- 1. bootstrap
    print(f"bootstrap: {args.n_boot} resamples x {len(probs)} models ...")
    boot = boot_metrics(y_te, probs, args.n_boot, SEED)
    rows = []
    for k in probs:
        pt = point_metrics(y_te, probs[k])
        for m in METRICS:
            lo, hi = pct(boot[k][m])
            rows.append(
                {
                    "model": k,
                    "metric": m,
                    "point": pt[m],
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci_width": hi - lo,
                }
            )
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(RESULTS_DIR / "model_eval_ci.csv", index=False)
    out.append(f"## Bootstrap 95% CI ({args.n_boot} resamples, test n={N})")
    out.append(md(ci_df))
    out.append(
        "(CI = test-set sampling noise only. Training randomness is in section 5.)\n"
    )

    # paired differences
    drows = []
    for base in ("lr_onehot", "lr_ordinal", "majority_prior"):
        for m in ("accuracy", "roc_auc", "pr_auc"):
            d = boot["gbm"][m] - boot[base][m]
            lo, hi = pct(d)
            drows.append(
                {
                    "gbm_minus": base,
                    "metric": m,
                    "diff": float(d.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "excludes_0": "yes" if lo > 0 or hi < 0 else "NO",
                }
            )
    out.append("## Paired bootstrap: GBM minus baseline")
    out.append(md(pd.DataFrame(drows)))

    # ---- 3. calibration
    frac, mean_pred = calibration_curve(y_te, p_gbm, n_bins=10, strategy="uniform")
    bins = np.clip((p_gbm * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y_te[m].mean() - p_gbm[m].mean())
    brier = point_metrics(y_te, p_gbm)["brier"]
    brier_prior = float(np.mean((y_te - prior) ** 2))
    out.append("\n## Calibration (GBM)")
    out.append(
        f"Brier={brier:.4f}  prior-only Brier={brier_prior:.4f}  "
        f"Brier skill score={(1 - brier / brier_prior):.4f}  weighted ECE={ece:.4f}"
    )
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(5.5, 7), gridspec_kw={"height_ratios": [3, 1]}
    )
    a1.plot([0, 1], [0, 1], "k--", label="perfect")
    a1.plot(mean_pred, frac, marker="o", label="GBM")
    fo, mo = calibration_curve(y_te, p_lr_oh, n_bins=10, strategy="uniform")
    a1.plot(mo, fo, marker="s", label="LR one-hot")
    a1.set_xlabel("mean predicted P(>50K)")
    a1.set_ylabel("observed fraction >50K")
    a1.set_title("Reliability curve (test)")
    a1.legend()
    a2.hist(p_gbm, bins=20)
    a2.set_xlabel("predicted probability")
    a2.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_eval_calibration.png", dpi=140)
    plt.close(fig)

    # ---- 4. per-group
    print("per-group metrics ...")
    grp = []
    for attr in ("sex", "race"):
        vals = raw_test[attr].astype(str).str.strip().values
        grp.append(group_table(attr, vals, y_te, p_gbm, 500, SEED))
    grp_df = pd.concat(grp, ignore_index=True)
    grp_df.to_csv(RESULTS_DIR / "model_eval_groups.csv", index=False)
    out.append("\n## Per-group metrics (GBM, threshold 0.5)")
    out.append(
        "sex and race are INPUT features of this model, so these gaps are not "
        "'fairness through unawareness'. Group base rates also differ."
    )
    out.append(md(grp_df))
    for attr in ("sex", "race"):
        g = grp_df[(grp_df.attr == attr) & (grp_df.n >= 100)]
        if len(g) >= 2:
            out.append(
                f"{attr} (groups n>=100): TPR gap={g.tpr.max() - g.tpr.min():.3f}  "
                f"selection-rate ratio min/max={g.selection_rate.min() / g.selection_rate.max():.3f}  "
                f"accuracy gap={g.accuracy.max() - g.accuracy.min():.3f}"
            )

    # ---- 5. gates
    out.append("\n## Promotion gates vs noise")
    pt = point_metrics(y_te, p_gbm)
    gates = pd.DataFrame(
        [
            {
                "gate": "accuracy>=0.75",
                "threshold": GATE_ACC,
                "gbm_point": pt["accuracy"],
                "gbm_ci_lo": float(
                    ci_df[
                        (ci_df.model == "gbm") & (ci_df.metric == "accuracy")
                    ].ci_lo.iloc[0]
                ),
                "majority_baseline": maj_acc,
                "majority_passes": "YES" if maj_acc >= GATE_ACC else "no",
            },
            {
                "gate": "f1>=0.70",
                "threshold": GATE_F1,
                "gbm_point": pt["f1"],
                "gbm_ci_lo": float(
                    ci_df[(ci_df.model == "gbm") & (ci_df.metric == "f1")].ci_lo.iloc[0]
                ),
                "majority_baseline": 0.0,
                "majority_passes": "no",
            },
            {
                "gate": "roc_auc>=0.75",
                "threshold": GATE_AUC,
                "gbm_point": pt["roc_auc"],
                "gbm_ci_lo": float(
                    ci_df[
                        (ci_df.model == "gbm") & (ci_df.metric == "roc_auc")
                    ].ci_lo.iloc[0]
                ),
                "majority_baseline": 0.5,
                "majority_passes": "no",
            },
        ]
    )
    out.append(md(gates))
    out.append(
        "Note: evaluate.py also has an F1>=0.70 gate (not in your task list). "
        "If majority_passes=YES, that gate cannot reject a model that always "
        "predicts <=50K."
    )

    print(f"seed-noise gate analysis: {args.k_models} models ...")
    sg = seed_gate_analysis(
        X_tr, y_tr, X_te, y_te, args.k_models, args.n_boot_pairs, SEED
    )
    out.append(
        f"\nDelta gate = {GATE_DELTA} ROC-AUC. Models differ ONLY by random_state."
    )
    out.append(
        f"seed AUCs: {[round(a, 5) for a in sg['seed_aucs']]}\n"
        f"seed-to-seed AUC SD = {sg['seed_auc_sd']:.5f}  range = {sg['seed_auc_range']:.5f}\n"
        f"SD of pairwise delta = {sg['pair_delta_sd']:.5f}  "
        f"paired test-set noise SD (bootstrap) = {sg['paired_test_noise_sd']:.5f}\n"
        f"ordered pairs = {sg['ordered_pairs']}"
    )
    fp = pd.DataFrame(
        [
            {
                "gate": "current: delta >= 0.001",
                "false_promotion_rate": sg["false_promo_current_gate"],
            },
            {
                "gate": "paired bootstrap: CI lower bound > 0",
                "false_promotion_rate": sg["false_promo_paired_bootstrap_gate"],
            },
            {
                "gate": "bootstrap CI > 0 AND delta >= 0.001",
                "false_promotion_rate": sg["false_promo_bootstrap_plus_margin"],
            },
        ]
    )
    out.append(md(fp))
    out.append(
        "False promotion = an equivalent model (different seed only) replacing the "
        "incumbent. The paired bootstrap removes TEST-set noise but not TRAINING "
        "noise, so it can still promote seed luck; compare the two numbers."
    )

    # ---- 6. test size
    print("CI width vs test size ...")
    sizes = [s for s in (500, 1000, 2500, 5000, 10000) if s < N] + [N]
    cw = ci_width_vs_n(y_te, p_gbm, sizes, args.n_boot_size, SEED)
    cw.to_csv(RESULTS_DIR / "model_eval_ci_width_vs_n.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(cw.n_test, cw.acc_ci_width, marker="o", label="accuracy CI width")
    ax.loglog(cw.n_test, cw.auc_ci_width, marker="s", label="ROC-AUC CI width")
    ax.axhline(2 * GATE_DELTA, color="red", ls=":", label="2 x delta gate (0.002)")
    ax.set_xlabel("test set size")
    ax.set_ylabel("95% CI width")
    ax.legend()
    ax.set_title("CI width shrinks ~ 1/sqrt(n)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_eval_ci_width_vs_n.png", dpi=140)
    plt.close(fig)

    acc = pt["accuracy"]
    n_for_half_pp = int((1.96**2) * acc * (1 - acc) / (0.005**2))
    out.append("\n## Test size vs CI width")
    out.append(md(cw))
    out.append(
        f"Analytic: n needed for accuracy CI half-width +/-0.5 pp = ~{n_for_half_pp}."
    )

    # ---- dataset justification
    just = [
        "DATASET / TEST SIZE JUSTIFICATION (numbers measured by this script)",
        f"- UCI Adult Income after cleaning: {len(raw_train) + N} rows "
        f"({len(raw_train)} train / {N} test = {N / (len(raw_train) + N):.0%} test).",
        f"- Positive rate train={float((raw_train[TARGET] == POS).mean()):.4f}, test={float(y_te.mean()):.4f}: "
        "imbalanced, so accuracy alone is misleading (majority baseline "
        f"{maj_acc:.4f}); PR-AUC and ROC-AUC reported alongside.",
        f"- Test n={N} gives accuracy 95% CI width "
        f"{float(cw.acc_ci_width.iloc[-1]):.4f} and ROC-AUC CI width "
        f"{float(cw.auc_ci_width.iloc[-1]):.4f}.",
        f"- Smaller tests widen the CI: n={int(cw.n_test.iloc[0])} gives accuracy width "
        f"{float(cw.acc_ci_width.iloc[0]):.4f}.",
        "- Limits: one fixed split (no CV), tabular census extract, no timestamps "
        "(so no real temporal drift), test set reused for gate decisions and "
        "reporting, synthetic drift only.",
    ]
    out.append("\n" + "\n".join(just))
    (RESULTS_DIR / "model_eval_dataset_justification.txt").write_text(
        "\n".join(just), encoding="utf-8"
    )

    text = "\n".join(out)
    (RESULTS_DIR / "model_eval_summary.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "model_eval.json").write_text(
        json.dumps(
            {"seed_gate": sg, "ece": ece, "brier": brier, "brier_prior": brier_prior},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + text)
    print(f"\nSaved to {RESULTS_DIR}")

    # ---- MLflow (fail-safe)
    try:
        from experiments.mlflow_utils import log_run, setup_tracking

        setup_tracking("task3_model_eval")
        metrics = {}
        for _, r in ci_df[ci_df.model == "gbm"].iterrows():
            metrics[f"gbm_{r.metric}"] = float(r.point)
            metrics[f"gbm_{r.metric}_ci_lo"] = float(r.ci_lo)
            metrics[f"gbm_{r.metric}_ci_hi"] = float(r.ci_hi)
        metrics.update(
            {
                "ece": float(ece),
                "seed_auc_sd": sg["seed_auc_sd"],
                "false_promo_current": sg["false_promo_current_gate"],
                "false_promo_bootstrap": sg["false_promo_paired_bootstrap_gate"],
            }
        )
        log_run(
            params={"n_boot": args.n_boot, "k_models": args.k_models, "n_test": N},
            metrics=metrics,
            artifacts=[str(p) for p in RESULTS_DIR.glob("model_eval*")],
            tags={"task": "3", "script": "model_eval"},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] skipped: {exc}")


if __name__ == "__main__":
    main()
