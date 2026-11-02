# SentinelML

ML monitoring and auto-retraining stack for a GradientBoostingClassifier on UCI Adult Income: model training, inference API, feature drift detection, data quality checks, automated retraining, and full observability via Prometheus + Grafana.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│              (DB / CSV / S3 via data/ingest.py)                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │   Airflow: training_dag     │  weekly
              │  ingest→validate→features  │
              │  →train→evaluate→register  │
              └─────────────┬──────────────┘
                            │ artifacts/
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    FastAPI  :8000                              │
│   POST /predict   POST /predict/batch   GET /metrics          │
│   api/predict.py  api/logger.py  api/metrics.py               │
└───────────┬───────────────────────────────┬───────────────────┘
            │ prediction logs               │ /metrics
            ▼                               ▼
      Postgres :5432              Prometheus Exporter :9100
      public.prediction_logs      (drift + quality + API metrics)
      drift_reports                          │
      quality_reports                        ▼
      model_registry                  Prometheus :9090
      alert_log                              │
            │                               ▼
            └──────────────────────► Grafana :3000
                                     (dashboards + alerts)
            ┌──────────────────────────────────────┐
            │   Airflow: drift_check_dag  (6h)     │
            │   load_ref→fetch_live→drift→quality  │
            │   →gate (drift, quality in parallel)│
            └──────────────┬───────────────────────┘
                           │ sets Airflow Variable
            ┌──────────────▼───────────────────────┐
            │  Airflow: retrain_trigger_dag (6h+30m)│
            │  check cooldown → trigger training   │
            └──────────────────────────────────────┘
```

Note: drift and quality reports are JSON files under `artifacts/`, not Postgres rows. Prediction logs are written by `api/logger.py` to `public.prediction_logs`; `ml.prediction_logs` in `init_db.sql` is currently unused.

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/Kushagra651/SentinelML
cd SentinelML
cp .env.example .env
# Edit .env — fill in POSTGRES_PASSWORD, SLACK_WEBHOOK_URL, etc.

# 2. Start the stack
docker compose up -d

# 3. Verify services
docker compose ps
curl http://localhost:8000/health      # API
curl http://localhost:9090/-/healthy   # Prometheus
open http://localhost:3000             # Grafana (admin / see .env)
open http://localhost:8080             # Airflow  (admin / see .env)
```

### Generate Airflow Fernet key
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Paste output into AIRFLOW_FERNET_KEY in .env
```

---

## Project Structure

```
SentinelML/
├── api/
│   ├── schemas.py          # PredictionInput / PredictionOutput / HealthResponse (Pydantic v2)
│   ├── predict.py          # Model loading, hot-reload, predict() / predict_batch()
│   ├── logger.py           # Async prediction logger → Postgres + JSONL
│   ├── metrics.py          # In-process metrics registry (thread-safe singleton)
│   └── main.py             # FastAPI app: /predict /metrics /logs /model/reload
├── data/
│   ├── ingest.py           # Pull raw DataFrame from DB / CSV / S3
│   ├── validate.py         # 5 hard + 2 soft checks → ValidationReport
│   ├── features.py         # FeaturePipeline fit/transform/save/load
│   └── drift_injector.py   # 6 active drift injectors + 1 no-op (temporal) for testing
├── training/
│   ├── train.py            # 7-step pipeline → model + pipeline artifacts
│   ├── evaluate.py         # Held-out eval: metrics, confusion matrix, calibration
│   └── register_model.py   # gates + ROC-AUC delta check, production_model.json pointer, registry JSON, audit log
├── monitoring/
│   ├── drift_report.py     # KS / Chi2 / PSI per feature + prediction drift
│   ├── quality_report.py   # 7 quality checks → QualityReport
│   └── prometheus_exporter.py  # HTTP server exporting all metrics as Prometheus text
├── airflow/dags/
│   ├── training_dag.py         # Weekly full training pipeline
│   ├── drift_check_dag.py      # 6-hourly drift + quality check
│   └── retrain_trigger_dag.py  # Polls drift Variable, triggers training w/ cooldown
├── alerting/
│   ├── notify.py               # Slack / PagerDuty / email / webhook dispatcher
│   └── grafana_alerts.json     # 11 rules (NOT mounted: provisioning this file crashes Grafana)
├── agent/                      # LangGraph agent: tools, graph, prompts, query_logger
├── experiments/                # measured experiments + MLflow helper (outputs in results/)
├── tests/
├── streamlit_app.py            # Streamlit dashboard (5 pages)
├── dockerfiles/
│   ├── Dockerfile.api          # Multi-stage FastAPI image
│   ├── Dockerfile.airflow      # Extends official Airflow image
│   └── Dockerfile.mlflow       # MLflow tracking server (own image: SQLAlchemy 2.x)
├── grafana/
│   ├── datasources.yaml        # Auto-provisioned Prometheus datasource
│   └── dashboards.json         # ML Monitoring dashboard (4 rows, 20+ panels)
├── prometheus/
│   └── prometheus.yml          # Scrape config: API, exporter, Airflow, node, cAdvisor
├── .github/workflows/
│   └── ci.yml                  # CI: ruff/black, mypy, pytest, image builds, GHCR push on main
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── init_db.sql                 # Postgres schema (7 tables in schema ml; api/logger.py also creates public.prediction_logs)
└── README.md
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness — returns model version |
| `GET` | `/ready` | Readiness — 503 if model not loaded |
| `POST` | `/predict` | Single prediction |
| `GET` | `/model/info` | Loaded model metadata |
| `GET` | `/metrics` | Prometheus scrape endpoint |
| `GET` | `/metrics/summary` | Human-readable metrics snapshot |
| `POST` | `/model/reload` | Hot-reload model without restart |
| `GET` | `/logs` | Time-windowed prediction log query |
| `POST` | `/agent/query` | Natural-language agent, body `{"question": "...", "stream": false}` |

**Prediction request:**
```json
{
  "age": 34, "fnlwgt": 200000, "education_num": 13,
  "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40,
  "workclass": "Private", "education": "Bachelors", "marital_status": "Never-married",
  "occupation": "Craft-repair", "relationship": "Not-in-family", "race": "White",
  "sex": "Male", "native_country": "United-States"
}
```

**Prediction response:**
```json
{
  "prediction": 1,
  "probability_class_0": 0.13,
  "probability_class_1": 0.87,
  "confidence": 0.87,
  "model_version": "v20260809_080617",
  "model_alias": "production",
  "latency_ms": 4.2,
  "features_used": ["age", "fnlwgt", "..."],
  "warnings": [],
  "request_id": "<uuid>"
}
```

---

## Drift Detection

Runs every 6 hours via `drift_check_dag`. Compares the last `DRIFT_WINDOW_HOURS` of live predictions against the training reference dataset.

| Feature type | Test | Threshold |
|---|---|---|
| Numerical | Kolmogorov-Smirnov | p < `KS_P_THRESHOLD` (0.05) |
| Categorical | Chi-square | p < `CHI2_P_THRESHOLD` (0.05) |
| Both | PSI | ≥ 0.1 warning / ≥ 0.2 critical |
| Predictions | Chi-square on class dist | p < 0.05 |

If drift or quality failure is detected, `retrain_trigger_dag` fires `ml_training_pipeline` after a `RETRAIN_COOLDOWN_HOURS` (12h) cooldown and a `RETRAIN_MAX_PER_DAY` (2) daily cap.

**Measured behaviour** (`experiments/results/`): the rule "any of 14 features has p<0.05" false-alarms on 44-62% of drift-free windows (200 windows per cell) because 14 tests run per window; Bonferroni lowers this to 6.5% at 1,000 samples. The cooldown and daily cap limit the cost of those false alarms.

---

## Grafana Alerts

| Alert | Severity | Condition |
|---|---|---|
| Critical Feature Drift | critical | `drift_critical_count > 0` for 5m |
| Feature Drift Warning | warning | `drift_warning_count > 0` for 15m |
| Prediction Distribution Drift | warning | `drift_prediction == 1` for 10m |
| Drift Report Stale | warning | report age > 12h |
| Data Quality Hard Failure | critical | `quality_hard_failures > 0` for 5m |
| High Feature Missing Rate | warning | missing % > 15% for 10m |
| Quality Report Stale | warning | report age > 12h |
| High API Error Rate | critical | error rate > 5% for 5m |
| High API Latency (p99) | warning | p99 > 2000ms for 5m |
| Low Prediction Confidence | warning | mean confidence < 0.55 for 15m |
| No Production Model | critical | model_loaded < 1 for 2m |

Routing: `critical` → Slack + PagerDuty. `warning` → Slack only.

---

## Measured results

All numbers come from `experiments/` (details in `experiments/results/`).

| Item | Result |
|---|---|
| Held-out test accuracy / ROC-AUC (n=15,060) | 86.67% [86.10, 87.18] / 0.9212 [0.9164, 0.9257] |
| Severe synthetic drift | accuracy 86.67% -> 79.96%; retraining recovers to 85.35% |
| Serving, 1 uvicorn worker, same-machine client | ~48-52 req/s; p95 298 ms at 10 users, 2.5 s at 100 users; 0% errors |
| Detector false-alarm rate, W=1000 | 44% (current rule), 6.5% (Bonferroni) |

## Development

### Local setup (no Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && source .env  # or use python-dotenv

# Run training pipeline
python -m training.train

# Start API
uvicorn api.main:app --reload --port 8000

# Run drift check manually
python -c "
from monitoring.drift_report import compute_drift_report
import pandas as pd
ref = pd.read_parquet('data/feature_store/train.parquet')
cur = pd.read_parquet('artifacts/live_window.parquet')
report = compute_drift_report(ref, cur)
print(report.summary)
"

# Emit metrics once
python -m monitoring.prometheus_exporter --once
```

### Rebuild a single service
```bash
docker compose build ml_api
docker compose up -d --no-deps ml_api
```

### Tail logs
```bash
docker compose logs -f ml_api
docker compose logs -f airflow_scheduler
```

### Rollback model
```bash
# From inside the ml_api container or directly:
python -c "
from training.register_model import rollback
rollback('20231201')
"
# Then hot-reload the API:
curl -X POST http://localhost:8000/model/reload
```

---

## Environment Variables

See [`.env.example`](.env.example) for the full annotated list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `production` | Label attached to all metrics and alerts |
| `AUTO_PROMOTE` | `false` | Auto-promote model after training |
| `DRIFT_WINDOW_HOURS` | `6` | Live window size for drift check |
| `RETRAIN_COOLDOWN_HOURS` | `12` | Minimum hours between auto-retrains |
| `SLACK_WEBHOOK_URL` | — | Required for Slack alerts |
| `PAGERDUTY_ROUTING_KEY` | — | Required for PagerDuty escalation |
| `AIRFLOW_FERNET_KEY` | — | **Required** — generate before first start |

---

## License

MIT