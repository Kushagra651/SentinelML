"""
experiments/load_test.py
------------------------
TASK 4 (no Locust, stdlib only): closed-loop load test of POST /predict.

Each virtual user sends requests back-to-back (no think time) for --duration
seconds. Payloads are real rows from the test split. Reports p50/p95/p99
latency, throughput (req/s) and error rate at 10, 50 and 100 users.

WARNING: every request is written to ml.prediction_logs. Run this on a
disposable stack or clear the table afterwards, or the drift window is polluted.

Run:
    python -m experiments.load_test --duration 30
    python -m experiments.load_test --users 10 50 100 --duration 30 --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from data.ingest import load_from_feature_store

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_payloads(n: int) -> list[bytes]:
    df = load_from_feature_store("test").drop(columns=["income"], errors="ignore")
    df = df.sample(n=n, random_state=42)
    return [
        json.dumps(r, default=lambda o: o.item()).encode()
        for r in df.to_dict("records")
    ]


def post(url: str, body: bytes, timeout: float = 30.0) -> tuple[float, bool]:
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            ok = 200 <= r.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        ok = False
    return (time.perf_counter() - t0) * 1000.0, ok


def run_level(url: str, payloads: list[bytes], users: int, duration: float) -> dict:
    lat: list[float] = []
    errs = [0]
    lock = threading.Lock()
    stop_at = time.perf_counter() + duration

    def worker(seed: int) -> None:
        i = seed
        local, e = [], 0
        while time.perf_counter() < stop_at:
            ms, ok = post(url, payloads[i % len(payloads)])
            i += 1
            if ok:
                local.append(ms)
            else:
                e += 1
        with lock:
            lat.extend(local)
            errs[0] += e

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(u * 37,)) for u in range(users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    total = len(lat) + errs[0]
    a = np.array(lat) if lat else np.array([np.nan])
    return {
        "users": users,
        "requests": total,
        "errors": errs[0],
        "error_rate_pct": round(100 * errs[0] / total, 2) if total else 0.0,
        "rps": round(len(lat) / wall, 1),
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "p99_ms": round(float(np.percentile(a, 99)), 1),
        "max_ms": round(float(np.max(a)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    payloads = build_payloads(500)
    target = f"{args.url}/predict"
    _, ok = post(target, payloads[0])
    if not ok:
        raise SystemExit(f"POST {target} failed. Is ml_api up and a model loaded?")
    for i in range(args.warmup):
        post(target, payloads[i])

    rows = []
    for u in args.users:
        print(f"running {u} users for {args.duration:.0f}s ...")
        r = run_level(target, payloads, u, args.duration)
        print(r)
        rows.append(r)
        time.sleep(3)

    hdr = "| users | req | err % | req/s | p50 ms | p95 ms | p99 ms | max ms |"
    lines = [hdr, "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['users']} | {r['requests']} | {r['error_rate_pct']} | {r['rps']} | "
            f"{r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | {r['max_ms']} |"
        )
    text = "\n".join(lines)
    print("\n" + text)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "load_test.json").write_text(json.dumps(rows, indent=2))
    (RESULTS_DIR / "load_test.md").write_text(text)
    print(f"\nSaved to {RESULTS_DIR}")

    try:
        from experiments.mlflow_utils import log_run, setup_tracking

        setup_tracking("task4_load_test")
        m = {}
        for r in rows:
            for k in ("rps", "p50_ms", "p95_ms", "p99_ms", "error_rate_pct"):
                m[f"u{r['users']}_{k}"] = float(r[k])
        log_run(
            params={"duration_s": args.duration, "users": str(args.users)},
            metrics=m,
            artifacts=[str(RESULTS_DIR / "load_test.json")],
            tags={"task": "4"},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] skipped: {exc}")


if __name__ == "__main__":
    main()
