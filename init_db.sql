-- =============================================================================
-- init_db.sql
-- Runs automatically on first postgres container startup via
-- /docker-entrypoint-initdb.d/
-- Creates the ml schema and all application tables.
-- Safe to re-run (IF NOT EXISTS guards on everything).
-- =============================================================================

-- Also create the airflow database if it doesn't exist
-- (Airflow init handles its own schema, we just need the DB)
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
GRANT ALL PRIVILEGES ON DATABASE airflow TO mluser;



-- All application tables live in the ml schema
CREATE SCHEMA IF NOT EXISTS ml;
SET search_path TO ml, public;

-- ── prediction_logs ───────────────────────────────────────────────────────────
-- One row per prediction request. features stored as JSONB for schema flexibility.
CREATE TABLE IF NOT EXISTS ml.prediction_logs (
    id              BIGSERIAL PRIMARY KEY,
    request_id      UUID NOT NULL UNIQUE,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   TEXT NOT NULL,
    model_alias     TEXT NOT NULL DEFAULT 'production',
    features        JSONB NOT NULL,
    prediction      SMALLINT NOT NULL CHECK (prediction IN (0, 1)),
    probability_0   DOUBLE PRECISION NOT NULL,
    probability_1   DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    latency_ms      DOUBLE PRECISION,
    ground_truth    SMALLINT CHECK (ground_truth IN (0, 1)),
    warnings        JSONB,
    schema_version  TEXT NOT NULL DEFAULT '1.0'
);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp
    ON ml.prediction_logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version
    ON ml.prediction_logs (model_version);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_request_id
    ON ml.prediction_logs (request_id);

-- ── drift_reports ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.drift_reports (
    id              BIGSERIAL PRIMARY KEY,
    report_id       TEXT NOT NULL UNIQUE,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   TEXT NOT NULL,
    window_start    TIMESTAMPTZ NOT NULL,
    window_end      TIMESTAMPTZ NOT NULL,
    overall_drifted BOOLEAN NOT NULL,
    drift_rate_pct  DOUBLE PRECISION,
    critical_count  INTEGER DEFAULT 0,
    warning_count   INTEGER DEFAULT 0,
    summary         JSONB
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_generated_at
    ON ml.drift_reports (generated_at DESC);

-- ── drift_feature_results ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.drift_feature_results (
    id          BIGSERIAL PRIMARY KEY,
    report_id   TEXT NOT NULL REFERENCES ml.drift_reports(report_id) ON DELETE CASCADE,
    feature     TEXT NOT NULL,
    method      TEXT NOT NULL,
    statistic   DOUBLE PRECISION,
    p_value     DOUBLE PRECISION,
    psi         DOUBLE PRECISION,
    drifted     BOOLEAN NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'ok'
);

CREATE INDEX IF NOT EXISTS idx_drift_feature_results_report_id
    ON ml.drift_feature_results (report_id);

-- ── quality_reports ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.quality_reports (
    id              BIGSERIAL PRIMARY KEY,
    report_id       TEXT NOT NULL UNIQUE,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   TEXT NOT NULL,
    window_start    TIMESTAMPTZ NOT NULL,
    window_end      TIMESTAMPTZ NOT NULL,
    overall_passed  BOOLEAN NOT NULL,
    window_size     INTEGER NOT NULL,
    hard_failures   JSONB,
    soft_warnings   JSONB
);

CREATE INDEX IF NOT EXISTS idx_quality_reports_generated_at
    ON ml.quality_reports (generated_at DESC);

-- ── quality_checks ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.quality_checks (
    id          BIGSERIAL PRIMARY KEY,
    report_id   TEXT NOT NULL REFERENCES ml.quality_reports(report_id) ON DELETE CASCADE,
    check_name  TEXT NOT NULL,
    severity    TEXT NOT NULL DEFAULT 'soft',
    passed      BOOLEAN NOT NULL,
    detail      TEXT
);

CREATE INDEX IF NOT EXISTS idx_quality_checks_report_id
    ON ml.quality_checks (report_id);

-- ── model_registry ────────────────────────────────────────────────────────────
-- SQL mirror of model_registry.json for queryability. JSON file is authoritative.
CREATE TABLE IF NOT EXISTS ml.model_registry (
    id              BIGSERIAL PRIMARY KEY,
    version_tag     TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL DEFAULT 'registered',
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at     TIMESTAMPTZ,
    val_accuracy    DOUBLE PRECISION,
    roc_auc         DOUBLE PRECISION,
    f1              DOUBLE PRECISION,
    model_path      TEXT,
    pipeline_path   TEXT,
    metadata        JSONB
);

-- ── alert_log ─────────────────────────────────────────────────────────────────
-- Append-only audit trail of all alerts sent by notify.py.
CREATE TABLE IF NOT EXISTS ml.alert_log (
    id          BIGSERIAL PRIMARY KEY,
    sent_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    channel     TEXT NOT NULL,
    severity    TEXT NOT NULL,
    title       TEXT NOT NULL,
    message     TEXT,
    labels      JSONB,
    success     BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_alert_log_sent_at
    ON ml.alert_log (sent_at DESC);