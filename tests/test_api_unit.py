import io
import re
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as app_main


def _get_metric_value(metrics_text: str, metric_name: str, labels: dict[str, str]) -> float:
    label_part = ",".join([f'{key}="{value}"' for key, value in labels.items()])
    pattern = rf"^{re.escape(metric_name)}\{{{re.escape(label_part)}\}} ([0-9.eE+-]+)$"
    for line in metrics_text.splitlines():
        match = re.match(pattern, line)
        if match:
            return float(match.group(1))
    return 0.0


@pytest.fixture
def client():
    with TestClient(app_main.app) as test_client:
        yield test_client


@pytest.fixture
def valid_image_bytes():
    image = np.full((20, 20, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_livez(client):
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_not_ready(client):
    app_main.pipeline = None
    app_main.pipeline_ready = False

    response = client.get("/readyz")
    assert response.status_code == 503


def test_readyz_ready(client):
    app_main.pipeline = MagicMock()
    app_main.pipeline_ready = True

    response = client.get("/readyz")
    assert response.status_code == 200


def test_health(client):
    app_main.pipeline = MagicMock()
    app_main.pipeline_ready = True

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert "version" in payload


def test_predict_empty(client):
    response = client.post("/predict")
    assert response.status_code == 400


def test_predict_invalid_image(client):
    app_main.pipeline = MagicMock()
    app_main.pipeline_ready = True
    before_metrics = client.get("/metrics").text
    before_decode_failed = _get_metric_value(
        before_metrics, "liveness_errors_total", {"kind": "decode_failed"}
    )

    response = client.post(
        "/predict",
        files={"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 400
    after_metrics = client.get("/metrics").text
    after_decode_failed = _get_metric_value(
        after_metrics, "liveness_errors_total", {"kind": "decode_failed"}
    )
    assert after_decode_failed >= before_decode_failed + 1


def test_predict_calls_pipeline(client, valid_image_bytes):
    mock_pipeline = MagicMock()
    mock_pipeline.process_frame.return_value = {
        "status": "accepted",
        "message": "ok",
        "liveness": {"is_real": True, "final_score": 0.99},
    }
    app_main.pipeline = mock_pipeline
    app_main.pipeline_ready = True

    response = client.post(
        "/predict",
        files={"file": ("tiny.jpg", io.BytesIO(valid_image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["is_real"] is True
    assert data["liveness_score"] == 0.99
    mock_pipeline.process_frame.assert_called_once()


def test_predict_pipeline_exception(client, valid_image_bytes):
    mock_pipeline = MagicMock()
    mock_pipeline.process_frame.side_effect = RuntimeError("boom")
    app_main.pipeline = mock_pipeline
    app_main.pipeline_ready = True
    before_metrics = client.get("/metrics").text
    before_pipeline_exception = _get_metric_value(
        before_metrics, "liveness_errors_total", {"kind": "pipeline_exception"}
    )

    response = client.post(
        "/predict",
        files={"file": ("tiny.jpg", io.BytesIO(valid_image_bytes), "image/jpeg")},
    )
    assert response.status_code == 500
    after_metrics = client.get("/metrics").text
    after_pipeline_exception = _get_metric_value(
        after_metrics, "liveness_errors_total", {"kind": "pipeline_exception"}
    )
    assert after_pipeline_exception >= before_pipeline_exception + 1


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
