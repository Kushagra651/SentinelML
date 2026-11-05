"""
experiments/detector_eval.py
----------------------------
TASK 2: measure the drift detector in monitoring/drift_report.py.

Measures
--------
1. False positive rate (FPR): how often each decision rule fires on windows that
   have NO drift. Two sources:
     - "independent": windows sampled from the held-out TEST split (same
       distribution as the reference, but not part of it). This is the honest one.
     - "ref_subsample": windows sampled from the reference itself. Biased low
       (the window is part of the reference), shown only to expose that bias.
2. Detection rate by drift intensity (5 levels + 0 control) x window size.
3. Detection delay in samples (growing prefix of a drifted stream).
4. Decision rules compared on the same per-feature p-values:
     raw_p05          production rule (overall_drifted: any feature p < 0.05)
     bonferroni       any p < 0.05 / 14
     bh_fdr           Benjamini-Hochberg at q = 0.05
     ks_and_psi_crit  any feature with severity == "critical" (p<0.05 AND PSI>=0.2)
     psi_warn         any feature PSI >= 0.1 (no p-value)

Drift type: covariate drift via data/drift_injector.inject_covariate_drift
(intensity = fraction of rows shifted AND shift size in std units).
Intensities 0.1 / 0.3 / 0.6 are the same as mild / moderate / severe in Task 1.

Run (from repo root, venv active):
    python -m experiments.detector_eval --quick      # ~ minutes, sanity check
    python -m experiments.detector_eval              # full run
    python -m experiments.detector_eval --traffic_per_hour 120

Nothing in the existing code is modified. Reports are NOT saved (save=False).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.drift_injector import inject_covariate_drift
from data.ingest import load_from_feature_store
from monitoring.drift_report import (
    KS_P_THRESHOLD,
    MIN_SAMPLES,
    PSI_WARNING,
    compute_drift_report,
)

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("monitoring.drift_report").setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALPHA = 0.05
TARGET_COL = "income"

RULES = [
    "raw_p05",
    "bonferroni",
    "bh_fdr",
    "ks_and_psi_crit",
    "psi_warn",
    "bonf_and_psi01",
    "bh_and_psi01",
]
RULE_LABEL = {
    "raw_p05": "raw p<0.05 (current)",
    "bonferroni": "Bonferroni",
    "bh_fdr": "Benjamini-Hochberg",
    "ks_and_psi_crit": "p<0.05 AND PSI>=0.2",
    "psi_warn": "PSI>=0.1 only",
    "bonf_and_psi01": "Bonferroni AND PSI>=0.1",
    "bh_and_psi01": "BH AND PSI>=0.1",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, float(c - h)), min(1.0, float(c + h))


def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
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


def decide(rep) -> dict[str, bool]:
    """Apply every decision rule to ONE drift report (same p-values for all)."""
    fr = rep.feature_results
    p = np.array([r.p_value for r in fr], dtype=float)
    psi = np.array([(r.psi or 0.0) for r in fr], dtype=float)
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= (np.arange(1, m + 1) / m) * ALPHA
    bh_sig = np.zeros(m, dtype=bool)
    if ok.any():
        bh_sig[order[: np.max(np.where(ok)[0]) + 1]] = True
    bonf_sig = p < ALPHA / m
    gate = psi >= PSI_WARNING
    return {
        "raw_p05": bool(rep.overall_drifted),
        "bonferroni": bool(bonf_sig.any()),
        "bh_fdr": bool(bh_sig.any()),
        "ks_and_psi_crit": any(r.severity == "critical" for r in fr),
        "psi_warn": bool(gate.any()),
        "bonf_and_psi01": bool((bonf_sig & gate).any()),
        "bh_and_psi01": bool((bh_sig & gate).any()),
    }


def evaluate_window(ref: pd.DataFrame, win: pd.DataFrame):
    rep = compute_drift_report(ref, win, save=False)
    return decide(rep), [r.feature for r in rep.feature_results if r.drifted], rep


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


def fpr_experiment(ref, pools, sizes, n_windows, seed):
    rows, feat_rows = [], []
    for src, pool in pools.items():
        for n in sizes:
            counts = {r: 0 for r in RULES}
            fcount: Counter = Counter()
            for i in range(n_windows):
                win = pool.sample(n=n, random_state=seed + i * 7 + n)
                d, feats, _ = evaluate_window(ref, win.reset_index(drop=True))
                for r in RULES:
                    counts[r] += int(d[r])
                fcount.update(feats)
            for r in RULES:
                lo, hi = wilson(counts[r], n_windows)
                rows.append(
                    {
                        "source": src,
                        "window": n,
                        "rule": r,
                        "n_windows": n_windows,
                        "fires": counts[r],
                        "fpr": counts[r] / n_windows,
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
            for f, c in fcount.items():
                feat_rows.append(
                    {
                        "source": src,
                        "window": n,
                        "feature": f,
                        "raw_fire_rate": c / n_windows,
                    }
                )
            print(
                f"  FPR [{src:14s}] W={n:<5d} "
                + " ".join(f"{r}={counts[r] / n_windows:.2f}" for r in RULES)
            )
    return pd.DataFrame(rows), pd.DataFrame(feat_rows)


def detection_experiment(ref, pool, sizes, levels, reps, seed):
    rows = []
    for n in sizes:
        for lvl in levels:
            counts = {r: 0 for r in RULES}
            for i in range(reps):
                s = seed + 1000 * i + n
                win = pool.sample(n=n, random_state=s).reset_index(drop=True)
                if lvl > 0:
                    win = inject_covariate_drift(win, intensity=lvl, seed=s)
                d, _, _ = evaluate_window(ref, win)
                for r in RULES:
                    counts[r] += int(d[r])
            for r in RULES:
                lo, hi = wilson(counts[r], reps)
                rows.append(
                    {
                        "window": n,
                        "level": lvl,
                        "rule": r,
                        "reps": reps,
                        "detect_rate": counts[r] / reps,
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
            print(
                f"  DET W={n:<5d} level={lvl:<4} "
                + " ".join(f"{r}={counts[r] / reps:.2f}" for r in RULES)
            )
    return pd.DataFrame(rows)


def delay_experiment(ref, pool, levels, reps, prefixes, seed):
    rows = []
    for lvl in levels:
        first: dict[str, list[float]] = {r: [] for r in RULES}
        for i in range(reps):
            s = seed + 5000 + i
            stream = pool.sample(n=max(prefixes), random_state=s).reset_index(drop=True)
            stream = inject_covariate_drift(stream, intensity=lvl, seed=s)
            fired: dict[str, int] = {}
            for k in prefixes:
                d, _, _ = evaluate_window(ref, stream.iloc[:k])
                for r in RULES:
                    if d[r] and r not in fired:
                        fired[r] = k
                if len(fired) == len(RULES):
                    break
            for r in RULES:
                first[r].append(fired.get(r, np.nan))
        for r in RULES:
            arr = np.array(first[r], dtype=float)
            det = ~np.isnan(arr)
            rows.append(
                {
                    "level": lvl,
                    "rule": r,
                    "detected_pct": float(det.mean() * 100),
                    "median_delay_samples": (
                        float(np.median(arr[det])) if det.any() else np.nan
                    ),
                }
            )
        print(f"  DELAY level={lvl} done")
    return pd.DataFrame(rows)


def recommend(fpr_df, det_df, sizes):
    cands = []
    for r in RULES:
        for n in sizes:
            f = fpr_df[
                (fpr_df.source == "independent")
                & (fpr_df.rule == r)
                & (fpr_df.window == n)
            ]
            d3 = det_df[
                (det_df.rule == r) & (det_df.window == n) & (det_df.level == 0.3)
            ]
            d1 = det_df[
                (det_df.rule == r) & (det_df.window == n) & (det_df.level == 0.1)
            ]
            if f.empty or d3.empty or d1.empty:
                continue
            fpr, det3, det1 = (
                float(f.fpr.iloc[0]),
                float(d3.detect_rate.iloc[0]),
                float(d1.detect_rate.iloc[0]),
            )
            if fpr <= 0.05 and det3 >= 0.80:
                cands.append(
                    {
                        "window": n,
                        "rule": r,
                        "fpr": fpr,
                        "detect_0.3": det3,
                        "detect_0.1_(harmless)": det1,
                    }
                )
    cands.sort(key=lambda c: (c["window"], c["detect_0.1_(harmless)"]))
    return cands


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------


def plot_fpr(fpr_df, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ind = fpr_df[fpr_df.source == "independent"]
    for r in RULES:
        s = ind[ind.rule == r].sort_values("window")
        ax.plot(s.window, s.fpr, marker="o", label=RULE_LABEL[r])
    s = fpr_df[(fpr_df.source == "ref_subsample") & (fpr_df.rule == "raw_p05")]
    if not s.empty:
        s = s.sort_values("window")
        ax.plot(
            s.window, s.fpr, "k--", marker="x", label="raw p<0.05, sampled from ref"
        )
    ax.axhline(0.05, color="red", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("window size (samples)")
    ax.set_ylabel("false positive rate (no-drift windows)")
    ax.set_title("Detector false positive rate by window size")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_detection(det_df, sizes, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, n in zip(axes.ravel(), sizes):
        for r in RULES:
            s = det_df[(det_df.window == n) & (det_df.rule == r)].sort_values("level")
            ax.plot(s.level, s.detect_rate, marker="o", label=RULE_LABEL[r])
        ax.set_title(f"window = {n}")
        ax.set_xlabel("covariate drift intensity (0 = no drift)")
        ax.set_ylabel("detection rate")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer windows / reps")
    ap.add_argument(
        "--windows", type=int, default=200, help="no-drift windows per cell"
    )
    ap.add_argument("--reps", type=int, default=50, help="drifted windows per cell")
    ap.add_argument("--delay_reps", type=int, default=20)
    ap.add_argument("--sizes", type=int, nargs="+", default=[30, 100, 500, 1000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--traffic_per_hour",
        type=float,
        default=None,
        help="measured prediction requests/hour (for the 6h-interval justification)",
    )
    args = ap.parse_args()
    if args.quick:
        args.windows, args.reps, args.delay_reps = 40, 15, 6

    sizes = sorted(args.sizes)
    if min(sizes) < MIN_SAMPLES:
        raise SystemExit(f"window sizes must be >= DRIFT_MIN_SAMPLES ({MIN_SAMPLES})")
    levels = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6]
    prefixes = [30, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ref = load_from_feature_store("train").drop(columns=[TARGET_COL], errors="ignore")
    pool = load_from_feature_store("test").drop(columns=[TARGET_COL], errors="ignore")
    ref = ref.reset_index(drop=True)
    pool = pool.reset_index(drop=True)
    print(f"reference(train)={len(ref)} pool(test)={len(pool)} features={ref.shape[1]}")
    print(f"KS/chi2 p threshold={KS_P_THRESHOLD}  min_samples={MIN_SAMPLES}")

    t0 = time.perf_counter()
    _ = evaluate_window(ref, pool.sample(n=1000, random_state=1).reset_index(drop=True))
    per_call = time.perf_counter() - t0
    n_calls = (
        2 * len(sizes) * args.windows
        + len(sizes) * len(levels) * args.reps
        + 5 * args.delay_reps * 6
    )
    print(
        f"one report = {per_call:.2f}s -> estimated total ~{per_call * n_calls / 60:.1f} min"
    )

    m_feats = len(evaluate_window(ref, pool.head(100))[2].feature_results)
    print(f"features tested per window (m) = {m_feats}")
    print(
        f"THEORY: raw rule FPR if tests were independent = 1-(1-0.05)^{m_feats} "
        f"= {1 - 0.95**m_feats:.3f}; Bonferroni/BH should be ~0.05 or lower"
    )

    print("\n[1/3] False positive rate")
    fpr_df, feat_df = fpr_experiment(
        ref,
        {"independent": pool, "ref_subsample": ref},
        sizes,
        args.windows,
        args.seed,
    )

    print("\n[2/3] Detection rate by intensity x window")
    det_df = detection_experiment(ref, pool, sizes, levels, args.reps, args.seed)

    print("\n[3/3] Detection delay")
    delay_df = delay_experiment(
        ref, pool, [x for x in levels if x > 0], args.delay_reps, prefixes, args.seed
    )

    # ---- save raw tables
    fpr_df.to_csv(RESULTS_DIR / "detector_fpr.csv", index=False)
    feat_df.to_csv(RESULTS_DIR / "detector_fpr_by_feature.csv", index=False)
    det_df.to_csv(RESULTS_DIR / "detector_detection.csv", index=False)
    delay_df.to_csv(RESULTS_DIR / "detector_delay.csv", index=False)
    plot_fpr(fpr_df, RESULTS_DIR / "detector_fpr.png")
    plot_detection(det_df, sizes, RESULTS_DIR / "detector_detection.png")

    # ---- summary text
    out = []
    out.append("# Task 2: detector evaluation")
    out.append(
        f"reference={len(ref)} rows (train)  no-drift pool={len(pool)} rows (test)  "
        f"windows/cell={args.windows}  drifted reps/cell={args.reps}\n"
    )

    ind = fpr_df[fpr_df.source == "independent"].copy()
    ind["fpr_95ci"] = ind.apply(lambda r: f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
    piv = ind.pivot(index="window", columns="rule", values="fpr")[RULES].reset_index()
    out.append("## 1. FPR, independent no-drift windows (lower is better)")
    out.append(md(piv))
    ref_sub = fpr_df[(fpr_df.source == "ref_subsample") & (fpr_df.rule == "raw_p05")][
        ["window", "fpr"]
    ].rename(columns={"fpr": "raw_rule_FPR_sampled_from_ref"})
    out.append("\nSame raw rule when windows are sampled from the reference itself:")
    out.append(md(ref_sub))
    out.append(
        "(Lower than the independent number = window overlaps the reference, "
        "which hides false alarms.)\n"
    )

    top_f = (
        feat_df[feat_df.source == "independent"]
        .sort_values("raw_fire_rate", ascending=False)
        .groupby("window")
        .head(3)
    )
    out.append("## Features that fire most on no-drift windows (raw rule)")
    out.append(
        md(top_f.sort_values(["window", "raw_fire_rate"], ascending=[True, False]))
    )

    out.append("\n## 2. Detection rate by intensity (rows) and rule (cols), per window")
    for n in sizes:
        t = (
            det_df[det_df.window == n]
            .pivot(index="level", columns="rule", values="detect_rate")[RULES]
            .reset_index()
        )
        out.append(f"\nwindow = {n}  (level 0.0 = no drift = FPR on test pool)")
        out.append(md(t))

    out.append(
        "\n## 3. Detection delay (samples of drifted stream until the rule fires)"
    )
    dl = delay_df.copy()
    out.append(md(dl))
    out.append(
        "(median_delay_samples is over runs that fired within 2000 samples; "
        "detected_pct is the share that fired at all.)"
    )

    cands = recommend(fpr_df, det_df, sizes)
    out.append("\n## 4. Recommendation from the numbers")
    out.append("Criteria: FPR <= 0.05 AND detection at intensity 0.3 >= 0.80.")
    out.append(
        "Intensity 0.1 was harmless in Task 1 (-0.18 pp accuracy, retrain gained nothing), "
        "so detection at 0.1 is a wasted-retrain cost; lower is better."
    )
    if cands:
        out.append(md(pd.DataFrame(cands)))
        best = cands[0]
        out.append(f"\nPick: window={best['window']}, rule={RULE_LABEL[best['rule']]}.")
    else:
        out.append(
            "NO (rule, window) pair meets both criteria. Say so honestly: the "
            "current detector cannot be tuned into the target region with these "
            "sizes. Look at the tables above for the closest pair."
        )

    out.append("\n## 5. Is a 6-hour interval justified?")
    out.append(
        "From code: drift_check = '0 */6 * * *', window 6h, min rows 30; "
        "retrain_trigger = '30 */6 * * *' (30 min later), cooldown 12h, max 2/day."
    )
    out.append(
        "Worst-case time from drift onset to retrain START = 6h (wait for next "
        "check) + 0.5h (trigger offset) = 6.5h, plus training time."
    )
    out.append("Requests per hour needed for a 6h window to reach each size:")
    need = pd.DataFrame(
        {"window": sizes, "req_per_hour_needed": [round(w / 6, 1) for w in sizes]}
    )
    out.append(md(need, "{:.1f}"))
    if args.traffic_per_hour:
        exp = args.traffic_per_hour * 6
        out.append(
            f"Measured traffic = {args.traffic_per_hour:.1f} req/h -> a 6h window holds "
            f"~{exp:.0f} samples."
        )
    else:
        out.append(
            "Traffic is UNKNOWN to me. Re-run with --traffic_per_hour <measured> "
            "(count of rows in ml.prediction_logs per hour) to finish this section."
        )

    text = "\n".join(out)
    (RESULTS_DIR / "detector_eval_summary.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "detector_eval.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "recommendation": cands,
                "fpr": fpr_df.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("\n" + text)
    print(f"\nSaved to {RESULTS_DIR}")

    # ---- MLflow (fail-safe; never breaks the run)
    try:
        from experiments.mlflow_utils import log_run, setup_tracking

        setup_tracking("task2_detector_eval")
        metrics = {}
        for _, r in ind.iterrows():
            metrics[f"fpr_{r.rule}_w{int(r.window)}"] = float(r.fpr)
        for _, r in det_df[det_df.level == 0.3].iterrows():
            metrics[f"det0.3_{r.rule}_w{int(r.window)}"] = float(r.detect_rate)
        log_run(
            params={"windows": args.windows, "reps": args.reps, "sizes": str(sizes)},
            metrics=metrics,
            artifacts=[str(p) for p in RESULTS_DIR.glob("detector_*")],
            tags={"task": "2", "script": "detector_eval"},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] skipped: {exc}")


if __name__ == "__main__":
    main()
