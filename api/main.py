"""
api/main.py — FastAPI application

Matched exactly to actual public APIs:
  api/predict.py : predict(), reload_if_stale(), force_reload(), get_model_info(), PredictionResult
  api/logger.py  : log_prediction_from_result(), flush(), shutdown(), query_logs()
  api/metrics.py : record_prediction_from_result(), record_error(), to_prometheus_text(), get_snapshot()
  api/schemas.py : PredictionInput, PredictionOutput, HealthResponse
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from fastapi.responses import StreamingResponse
from agent.graph import run_agent, stream_agent
from agent.query_logger import log_query

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

import api.predict as predictor_module
from api.predict import (
    PredictionResult,
    force_reload,
    get_model_info,
    predict,
    reload_if_stale,
)
from api.logger import flush, log_prediction_from_result, query_logs, shutdown
from api.metrics import (
    get_snapshot,
    record_error,
    record_prediction_from_result,
    to_prometheus_text,
)
from api.schemas import HealthResponse, PredictionInput, PredictionOutput

log = logging.getLogger(__name__)
_START_TIME = time.time()


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting ML Monitoring API...")
    try:
        predictor_module._ensure_loaded()
        info = get_model_info()
        log.info(
            "Model loaded: version=%s alias=%s", info.get("version"), info.get("alias")
        )
    except FileNotFoundError as e:
        log.warning("Model not found at startup (run training first): %s", e)
    except Exception as e:
        log.error("Unexpected startup error: %s", e)

    yield

    log.info("Shutdown — flushing prediction logs...")
    flush()
    shutdown()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ML Pipeline Monitor — UCI Adult Income",
    version="1.0.0",
    description="Predicts whether income >$50K. Drift detection + Prometheus monitoring.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware — latency header + hot-reload check ────────────────────────────


@app.middleware("http")
async def track_latency(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    if request.url.path == "/predict":
        reload_if_stale()
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


# ── Health / readiness ────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    info = get_model_info()
    return HealthResponse(
        status="ok",
        model_loaded=info.get("status") == "loaded",
        model_version=info.get("version"),
        model_alias=info.get("alias"),
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@app.get("/ready", tags=["ops"])
def ready():
    info = get_model_info()
    if info.get("status") != "loaded":
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ready", "model_version": info.get("version")}


# ── Inference ─────────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictionOutput, tags=["inference"])
def predict_endpoint(req: PredictionInput):
    features = req.model_dump()
    request_id = str(uuid.uuid4())

    try:
        result: PredictionResult = predict(features)
    except (ValueError, RuntimeError) as e:
        record_error(kind="prediction_error")
        log.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    output = PredictionOutput(
        prediction=result.prediction,
        probability_class_0=result.probability_class_0,
        probability_class_1=result.probability_class_1,
        confidence=result.confidence,
        model_version=result.model_version,
        model_alias=result.model_alias,
        latency_ms=result.latency_ms,
        features_used=result.features_used,
        warnings=result.warnings,
        request_id=request_id,
    )

    record_prediction_from_result(result)

    try:
        log_prediction_from_result(
            features=features,
            result=result,
            request_id=request_id,
        )
    except Exception as e:
        log.warning("Prediction log enqueue failed (non-fatal): %s", e)

    return output


# ── Model ops ─────────────────────────────────────────────────────────────────


@app.post("/model/reload", tags=["ops"])
def reload_model():
    """Force hot-reload of model from registry without container restart."""
    try:
        force_reload()
        info = get_model_info()
        return {"reloaded": True, "model_version": info.get("version")}
    except Exception as e:
        log.error("Force reload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["ops"])
def model_info():
    """Metadata about the currently loaded model bundle."""
    return get_model_info()


# ── Monitoring ────────────────────────────────────────────────────────────────


@app.get("/metrics", tags=["monitoring"])
def prometheus_metrics():
    """Prometheus scrape endpoint (text/plain 0.0.4)."""
    return Response(
        content=to_prometheus_text(),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/metrics/summary", tags=["monitoring"])
def metrics_summary():
    """Human-readable metrics snapshot."""
    return get_snapshot().to_dict()


@app.get("/logs", tags=["monitoring"])
def get_logs(hours: int = 1, limit: int = 100):
    """Time-windowed prediction log retrieval."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    try:
        df = query_logs(start=start, end=now, limit=limit)
        records = df.to_dict(orient="records") if not df.empty else []
        return {"count": len(records), "logs": records}
    except Exception as e:
        log.error("query_logs failed: %s", e)
        return {"count": 0, "logs": [], "error": str(e)}


@app.post("/agent/query")
async def agent_query(body: dict):
    import time
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            stream_agent(question),
            media_type="text/event-stream",
        )

    t0 = time.perf_counter()
    answer = run_agent(question)
    latency = (time.perf_counter() - t0) * 1000

    model_info = get_model_info()
    log_query(
        question=question,
        answer=answer,
        latency_ms=round(latency, 2),
        model_version=model_info.get("version_tag"),
    )

    return {"question": question, "answer": answer, "latency_ms": round(latency, 2)}