import os
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.schemas import HealthResponse, PredictResponse
from src.pipeline.pipeline import FaceLivenessPipeline


APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
logger = logging.getLogger(__name__)

REQUEST_COUNTER = Counter(
    "liveness_requests_total",
    "Total number of liveness requests by status.",
    ["status"],
)
ERROR_COUNTER = Counter(
    "liveness_errors_total",
    "Total number of liveness errors by kind.",
    ["kind"],
)
LATENCY_HISTOGRAM = Histogram(
    "liveness_latency_seconds",
    "Latency of liveness prediction requests.",
)

pipeline: FaceLivenessPipeline | None = None
pipeline_ready = False


def _load_pipeline() -> FaceLivenessPipeline:
    config_path = Path("config/config.yaml")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return FaceLivenessPipeline(config["pipeline"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipeline
    global pipeline_ready

    try:
        pipeline = _load_pipeline()
        pipeline_ready = True
        logger.info("pipeline loaded version=%s", APP_VERSION)
    except Exception:
        pipeline = None
        pipeline_ready = False
        logger.exception("pipeline load failed")
    yield


app = FastAPI(title="Face Liveness API", version=APP_VERSION, lifespan=lifespan)


@app.get("/livez")
def livez() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    if not pipeline_ready or pipeline is None:
        raise HTTPException(status_code=503, detail="pipeline_not_ready")
    return {"status": "ok", "pipeline_ready": True}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    is_loaded = bool(pipeline_ready and pipeline is not None)
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        model_loaded=is_loaded,
        version=APP_VERSION,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile | None = File(None)) -> PredictResponse:
    if file is None:
        ERROR_COUNTER.labels(kind="empty_file").inc()
        raise HTTPException(status_code=400, detail="No file provided")

    if not pipeline_ready or pipeline is None:
        ERROR_COUNTER.labels(kind="pipeline_not_ready").inc()
        raise HTTPException(status_code=503, detail="Pipeline is not loaded")

    try:
        file_bytes = file.file.read()
    except Exception as exc:
        ERROR_COUNTER.labels(kind="read_failed").inc()
        raise HTTPException(status_code=400, detail="Failed to read upload") from exc

    if not file_bytes:
        ERROR_COUNTER.labels(kind="empty_file").inc()
        raise HTTPException(status_code=400, detail="File is empty")

    np_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        ERROR_COUNTER.labels(kind="decode_failed").inc()
        raise HTTPException(status_code=400, detail="Invalid image file")

    start_time = time.perf_counter()
    try:
        result = pipeline.process_frame(frame)
    except Exception as exc:
        ERROR_COUNTER.labels(kind="pipeline_exception").inc()
        REQUEST_COUNTER.labels(status="error").inc()
        raise HTTPException(status_code=500, detail="Pipeline processing failed") from exc

    latency_seconds = time.perf_counter() - start_time
    LATENCY_HISTOGRAM.observe(latency_seconds)

    status = result.get("status", "unknown")
    REQUEST_COUNTER.labels(status=status).inc()

    liveness_result = result.get("liveness", {})
    return PredictResponse(
        status=status,
        is_real=bool(liveness_result.get("is_real", False)),
        liveness_score=float(liveness_result.get("final_score", 0.0)),
        message=str(result.get("message", "")),
        latency_ms=latency_seconds * 1000.0,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
