from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as app_main


def _resolve_fixture_path(filename: str) -> Path:
    fixture_path = Path("tests/fixtures") / filename
    if fixture_path.exists():
        return fixture_path
    return Path("data/test") / ("normal" if filename == "tiny.jpg" else "spoof") / (
        "1490_4.jpg" if filename == "tiny.jpg" else "151_4.jpg"
    )


@pytest.fixture(scope="module")
def integration_client():
    previous_pipeline = app_main.pipeline
    previous_ready = app_main.pipeline_ready
    try:
        app_main.pipeline = app_main._load_pipeline()
        app_main.pipeline_ready = True
    except Exception as exc:
        pytest.skip(f"Integration pipeline unavailable: {exc}")

    with TestClient(app_main.app) as client:
        yield client

    app_main.pipeline = previous_pipeline
    app_main.pipeline_ready = previous_ready


@pytest.mark.integration
def test_predict_real_face(integration_client):
    real_image = _resolve_fixture_path("tiny.jpg")
    if not real_image.exists():
        pytest.skip("Real integration fixture image not found")

    with real_image.open("rb") as image_file:
        response = integration_client.post(
            "/predict",
            files={"file": (real_image.name, image_file, "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"accepted", "rejected"}
    assert isinstance(payload["is_real"], bool)
    assert isinstance(payload["liveness_score"], float)


@pytest.mark.integration
def test_predict_spoof_image(integration_client):
    spoof_image = _resolve_fixture_path("spoof.jpg")
    if not spoof_image.exists():
        pytest.skip("Spoof integration fixture image not found")

    with spoof_image.open("rb") as image_file:
        response = integration_client.post(
            "/predict",
            files={"file": (spoof_image.name, image_file, "image/jpeg")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"accepted", "rejected"}
    assert isinstance(payload["is_real"], bool)
    assert isinstance(payload["liveness_score"], float)
