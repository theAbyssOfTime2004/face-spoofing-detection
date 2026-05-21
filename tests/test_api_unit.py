import io
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as app_main


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

    response = client.post(
        "/predict",
        files={"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 400


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

    response = client.post(
        "/predict",
        files={"file": ("tiny.jpg", io.BytesIO(valid_image_bytes), "image/jpeg")},
    )
    assert response.status_code == 500


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
