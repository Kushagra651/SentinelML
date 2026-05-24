#!/bin/bash
# ── .devcontainer/post-start.sh ──────────────────────────────────────────────
# Runs on the HOST side after all containers start.
# Waits for each service to be healthy before printing the access URLs.
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "⏳ Waiting for Postgres..."
until docker exec ml_postgres pg_isready -U "${POSTGRES_USER:-mluser}" 2>/dev/null; do
  sleep 2
done
echo "✅ Postgres ready"

echo "⏳ Waiting for FastAPI..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
  sleep 3
done
echo "✅ FastAPI ready"

echo "⏳ Waiting for Airflow webserver..."
until curl -sf http://localhost:8080/health > /dev/null 2>&1; do
  sleep 5
done
echo "✅ Airflow ready"

echo "⏳ Waiting for Grafana..."
until curl -sf http://localhost:3000/api/health > /dev/null 2>&1; do
  sleep 3
done
echo "✅ Grafana ready"

echo ""
echo "🚀 All services are up!"
echo "──────────────────────────────────────────────────────"
echo "  FastAPI docs    → http://localhost:8000/docs"
echo "  Prom exporter   → http://localhost:9100/metrics"
echo "  Airflow UI      → http://localhost:8080  (admin / admin)"
echo "  Prometheus      → http://localhost:9090"
echo "  Grafana         → http://localhost:3000  (admin / admin)"
echo "  cAdvisor        → http://localhost:8081"
echo "  Node Exporter   → http://localhost:9101/metrics"
echo "──────────────────────────────────────────────────────"