"""
apply_task7_fixes.py  (run from the repo root)

  python apply_task7_fixes.py --dry-run     # report only
  python apply_task7_fixes.py               # apply

Every edit is an exact find/replace that must match exactly once. A file is only
written if ALL of its edits match; otherwise it is left untouched and reported.
"""
import argparse
import sys
from pathlib import Path

EDITS = {}

# ---------------------------------------------------------------- README.md
EDITS["README.md"] = [
    ("# ML Monitoring Pipeline\n\nProduction-grade ML monitoring stack:",
     "# SentinelML\n\nML monitoring and auto-retraining stack for a GradientBoostingClassifier on UCI Adult Income:"),
    ("(6h+15m)", "(6h+30m)"),
    ("→push_metrics→gate                 ", "→gate (drift, quality in parallel)"),
    ("      prediction_logs             (drift + quality + API metrics)",
     "      public.prediction_logs      (drift + quality + API metrics)"),
    ("---\n\n## Quick Start",
     "Note: drift and quality reports are JSON files under `artifacts/`, not Postgres rows. "
     "Prediction logs are written by `api/logger.py` to `public.prediction_logs`; "
     "`ml.prediction_logs` in `init_db.sql` is currently unused.\n\n---\n\n## Quick Start"),
    ("git clone <repo>\ncd ml_monitoring", "git clone https://github.com/Kushagra651/SentinelML\ncd SentinelML"),
    ("ml_monitoring/\n├── api/", "SentinelML/\n├── api/"),
    ("# FeatureSchema — single source of truth for columns/dtypes",
     "# PredictionInput / PredictionOutput / HealthResponse (Pydantic v2)"),
    ("# 7 synthetic drift types for testing",
     "# 6 active drift injectors + 1 no-op (temporal) for testing"),
    ("# 3-gate promotion, registry JSON, symlinks, audit log",
     "# gates + ROC-AUC delta check, production_model.json pointer, registry JSON, audit log"),
    ("# 11 Grafana Unified Alerting rules",
     "# 11 rules (NOT mounted: provisioning this file crashes Grafana)"),
    ("├── dockerfiles/",
     "├── agent/                      # LangGraph agent: tools, graph, prompts, query_logger\n"
     "├── experiments/                # measured experiments + MLflow helper (outputs in results/)\n"
     "├── tests/\n├── streamlit_app.py            # Streamlit dashboard (5 pages)\n├── dockerfiles/"),
    ("│   └── Dockerfile.airflow      # Extends official Airflow image",
     "│   ├── Dockerfile.airflow      # Extends official Airflow image\n"
     "│   └── Dockerfile.mlflow       # MLflow tracking server (own image: SQLAlchemy 2.x)"),
    ("# CI: lint + test on push", "# CI: ruff/black, mypy, pytest, image builds, GHCR push on main"),
    ("# Postgres schema (6 tables)",
     "# Postgres schema (7 tables in schema ml; api/logger.py also creates public.prediction_logs)"),
    ("| `POST` | `/predict/batch` | Batch predictions |\n",
     "| `GET` | `/model/info` | Loaded model metadata |\n"),
    ("| `GET` | `/logs` | Time-windowed prediction log query |",
     "| `GET` | `/logs` | Time-windowed prediction log query |\n"
     "| `POST` | `/agent/query` | Natural-language agent, body `{\"question\": \"...\", \"stream\": false}` |"),
    ('{\n  "request_id": "req-123",\n  "features": { "feature_1": 1.5, "feature_2": "cat_a" }\n}',
     '{\n  "age": 34, "fnlwgt": 200000, "education_num": 13,\n  "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40,\n'
     '  "workclass": "Private", "education": "Bachelors", "marital_status": "Never-married",\n'
     '  "occupation": "Craft-repair", "relationship": "Not-in-family", "race": "White",\n'
     '  "sex": "Male", "native_country": "United-States"\n}'),
    ('{\n  "request_id": "req-123",\n  "predicted_class": "1",\n  "confidence": 0.87,\n  "probabilities": { "0": 0.13, "1": 0.87 },\n  "model_version": "20240101"\n}',
     '{\n  "prediction": 1,\n  "probability_class_0": 0.13,\n  "probability_class_1": 0.87,\n  "confidence": 0.87,\n'
     '  "model_version": "v20260809_080617",\n  "model_alias": "production",\n  "latency_ms": 4.2,\n'
     '  "features_used": ["age", "fnlwgt", "..."],\n  "warnings": [],\n  "request_id": "<uuid>"\n}'),
    ("after a `RETRAIN_COOLDOWN_HOURS` (12h) guard.",
     "after a `RETRAIN_COOLDOWN_HOURS` (12h) cooldown and a `RETRAIN_MAX_PER_DAY` (2) daily cap.\n\n"
     "**Measured behaviour** (`experiments/results/`): the rule \"any of 14 features has p<0.05\" false-alarms on "
     "44-62% of drift-free windows (200 windows per cell) because 14 tests run per window; Bonferroni lowers this to "
     "6.5% at 1,000 samples. The cooldown and daily cap limit the cost of those false alarms."),
    ("## Development\n\n### Local setup",
     "## Measured results\n\nAll numbers come from `experiments/` (details in `experiments/results/`).\n\n"
     "| Item | Result |\n|---|---|\n"
     "| Held-out test accuracy / ROC-AUC (n=15,060) | 86.67% [86.10, 87.18] / 0.9212 [0.9164, 0.9257] |\n"
     "| Severe synthetic drift | accuracy 86.67% -> 79.96%; retraining recovers to 85.35% |\n"
     "| Serving, 1 uvicorn worker, same-machine client | ~48-52 req/s; p95 298 ms at 10 users, 2.5 s at 100 users; 0% errors |\n"
     "| Detector false-alarm rate, W=1000 | 44% (current rule), 6.5% (Bonferroni) |\n\n"
     "## Development\n\n### Local setup"),
    ("artifacts/features_<tag>.parquet", "data/feature_store/train.parquet"),
    ("from training.register_model import rollback_model\nrollback_model(to_version='20231201')",
     "from training.register_model import rollback\nrollback('20231201')"),
    ("docker compose logs -f airflow-scheduler", "docker compose logs -f airflow_scheduler"),
]

# ------------------------------------------------------------- .env.example
EDITS[".env.example"] = [
    ("WORKERS=2", "# WORKERS is not read by any code: Dockerfile.api hard-codes uvicorn --workers 1"),
    ("LOG_BUFFER_SIZE=500", "LOG_BUFFER_SIZE=200"),
    ("LOG_FLUSH_INTERVAL_SEC=30", "LOG_FLUSH_INTERVAL=5"),
    ("LOG_JSONL_PATH=/app/artifacts/logs", "LOG_DIR=logs"),
    ("AIRFLOW_ADMIN_USER=admin\nAIRFLOW_ADMIN_PASSWORD=admin", "AIRFLOW_USER=admin\nAIRFLOW_PASSWORD=admin"),
    ("GRAFANA_ADMIN_USER=admin\nGRAFANA_ADMIN_PASSWORD=admin", "GRAFANA_USER=admin\nGRAFANA_PASSWORD=admin"),
]

# ------------------------------------------------------- docker-compose.yml
EDITS["docker-compose.yml"] = [
    ('version: "3.9"\n\n', ""),
    ("    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}\n    TARGET_COLUMN: income",
     "    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}\n"
     "    DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@ml_postgres:5432/${POSTGRES_DB}\n"
     "    TARGET_COLUMN: income"),
    ("      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}\n      LOG_LEVEL:",
     "      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}\n"
     "      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@ml_postgres:5432/${POSTGRES_DB}\n"
     "      LOG_LEVEL:"),
]

# --------------------------------------------------------------- init_db.sql
EDITS["init_db.sql"] = [
    ("-- One row per prediction request. features stored as JSONB for schema flexibility.",
     "-- One row per prediction request. features stored as JSONB for schema flexibility.\n"
     "-- NOTE: currently UNUSED. api/logger.py creates and writes public.prediction_logs (TEXT columns)."),
]

# ------------------------------------------------------------ agent/tools.py
EDITS["agent/tools.py"] = [
    ("  get_model_info()            → GET /model/info\n  get_metrics_snapshot()      → GET /metrics/summary\n  get_prediction_logs(hours)  → GET /logs?hours=N",
     "  get_model_info()            → reads artifacts/production_model.json\n  get_metrics_snapshot()      → api.metrics.get_snapshot() (in-process)\n  get_prediction_logs(hours)  → api.logger.query_logs() (in-process)"),
    ('_HTTP_TIMEOUT = int(os.getenv("AGENT_HTTP_TIMEOUT", "10"))\n',
     '_HTTP_TIMEOUT = int(os.getenv("AGENT_HTTP_TIMEOUT", "10"))\n_MAX_LOG_ROWS = 50_000  # same cap the drift DAG uses; counts above it are flagged "truncated"\n'),
    ('summary.get("total_features", 0)', 'summary.get("total_features_checked", 0)'),
    ("    Calls GET /logs?hours={hours} on the live FastAPI service.",
     "    Queries api.logger.query_logs() in-process (an HTTP call back into the same\n    single-worker server deadlocks)."),
    ("df = query_logs(start, end, limit=500)", "df = query_logs(start, end, limit=_MAX_LOG_ROWS)"),
    ('        "total_predictions": len(records),\n',
     '        "total_predictions": len(records),\n        "truncated": len(records) >= _MAX_LOG_ROWS,\n'),
    ("            for r in records[:10]\n", "            for r in records[-10:][::-1]\n"),
]

# --------------------------------------------------------------- api/main.py
EDITS["api/main.py"] = [
    ('@app.post("/agent/query")\nasync def agent_query(body: dict):\n    import time\n\n    question',
     '@app.post("/agent/query")\ndef agent_query(body: dict):\n    question'),
    ('model_version=model_info.get("version_tag"),', 'model_version=model_info.get("version"),'),
]

TEST_FILE = '''"""tests/test_agent_tools.py: regression tests for two agent tool bugs."""
import pandas as pd

import agent.tools as tools


def test_drift_report_reports_total_features(monkeypatch):
    report = {
        "overall_drifted": True,
        "drifted_features": ["age"],
        "summary": {"total_features_checked": 14, "critical_count": 1},
        "feature_results": [{"feature": "age", "psi": 0.3, "severity": "critical", "drifted": True}],
    }
    monkeypatch.setattr(tools, "_latest_json", lambda subdir: dict(report))
    out = tools.get_drift_report.invoke({})
    assert out["total_features_checked"] == 14


def test_prediction_logs_counts_all_rows_and_samples_newest(monkeypatch):
    import api.logger as lg

    n = 600  # more than the old 500-row cap
    df = pd.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(n)],
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
            "prediction": [i % 2 for i in range(n)],
            "confidence": [0.9] * n,
            "model_version": ["v1"] * n,
            "ground_truth": [None] * n,
        }
    )
    monkeypatch.setattr(lg, "query_logs", lambda start, end, limit=100_000, **kw: df)
    out = tools.get_prediction_logs.invoke({"hours": 6})
    assert out["total_predictions"] == n
    assert out["recent_sample"][0]["request_id"] == "r599"  # newest first
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    failed = 0
    for rel, edits in EDITS.items():
        p = root / rel
        if not p.exists():
            print(f"SKIP  {rel}: not found")
            failed += 1
            continue
        raw = p.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        bad = []
        for i, (old, new) in enumerate(edits, 1):
            c = text.count(old)
            if c != 1:
                bad.append((i, c, old[:60].replace("\n", "\\n")))
            else:
                text = text.replace(old, new)
        if bad:
            failed += 1
            print(f"FAIL  {rel}: untouched")
            for i, c, snip in bad:
                print(f"      edit {i}: matched {c}x  -> {snip!r}")
            continue
        if not args.dry_run:
            p.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
        print(f"{'OK(dry)' if args.dry_run else 'DONE'}  {rel}: {len(edits)} edits")
    t = root / "tests" / "test_agent_tools.py"
    if not args.dry_run and not t.exists():
        t.parent.mkdir(exist_ok=True)
        t.write_text(TEST_FILE, encoding="utf-8")
        print("DONE  tests/test_agent_tools.py created")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()