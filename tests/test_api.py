"""API tests with a tiny real model (no MLflow): statuses, errors and security."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from neuroscan.config import CLASS_NAMES, load_config
from neuroscan.models import build_model
from neuroscan.serving.api import ServedModel, create_app
from neuroscan.serving.packaged import NeuroScanModel, serving_config


def png(arr, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


GRAY = png(np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1)))
PHOTO = png(np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def served():
    model = build_model(load_config("configs/cnn_gap.yaml", {"data.image_size": 64}))
    predictor = NeuroScanModel()
    predictor.setup(model, serving_config(1.5, 64, "1", uncertain_threshold=0.0))
    return ServedModel(predictor, "models:/test@champion", "7", {"test_macro_f1": 0.95})


@pytest.fixture
def client(served, monkeypatch):
    monkeypatch.setenv("NEUROSCAN_MAX_UPLOAD_MB", "1")
    with TestClient(create_app(loader=lambda uri: served)) as c:
        yield c


def test_health_ready_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["model_version"] == "7"
    info = client.get("/v1/info").json()
    assert info["classes"] == list(CLASS_NAMES)
    assert info["temperature"] == 1.5 and info["metrics"]["test_macro_f1"] == 0.95


def test_predict_ok_with_heatmap(client):
    r = client.post("/v1/predict", files={"file": ("scan.png", GRAY, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_version"] == "7"
    assert body["prediction"] in CLASS_NAMES
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert body["heatmap_png_base64"] and body["input_png_base64"]
    assert "not a medical device" in body["disclaimer"]


def test_color_photo_is_rejected_not_answered(client):
    body = client.post("/v1/predict", files={"file": ("cat.jpg", PHOTO, "image/jpeg")}).json()
    assert body["status"] == "rejected_input" and body["reason"] == "not_grayscale"
    assert body["prediction"] is None and body["heatmap_png_base64"] is None


def test_text_file_renamed_jpg_is_400(client):
    r = client.post("/v1/predict", files={"file": ("x.jpg", b"hello, not an image", "image/jpeg")})
    assert r.status_code == 400 and r.json() == {"error": "not_an_image", "detail": None}


def test_too_large_is_413(client):
    big = b"\xff" * (1024 * 1024 + 10)
    r = client.post("/v1/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413 and r.json()["error"] == "file_too_large"


def test_path_traversal_filename_is_harmless(client, tmp_path):
    r = client.post("/v1/predict", files={"file": ("../../etc/passwd", GRAY, "image/png")})
    assert r.status_code == 200  # the filename is never used


def test_batch(client):
    files = [("files", ("a.png", GRAY, "image/png")), ("files", ("b.png", PHOTO, "image/png"))]
    body = client.post("/v1/predict/batch", files=files).json()
    assert [x["status"] for x in body["results"]] == ["ok", "rejected_input"]


def test_security_headers_and_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "NeuroScan" in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"
    assert "content-security-policy" not in client.get("/docs").headers
    assert client.get("/static/app.js").status_code == 200


def test_not_ready_when_model_fails(monkeypatch):
    def broken(uri):
        raise RuntimeError("registry unreachable")

    with TestClient(create_app(loader=broken)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        r = c.post("/v1/predict", files={"file": ("a.png", GRAY, "image/png")})
        assert r.status_code == 503 and r.json()["error"] == "model_not_loaded"


def test_metrics_and_outcome_log(served, monkeypatch, tmp_path):
    import json

    log = tmp_path / "predictions.jsonl"
    monkeypatch.setenv("NEUROSCAN_PREDICTION_LOG", str(log))
    with TestClient(create_app(loader=lambda uri: served)) as c:
        c.post("/v1/predict", files={"file": ("a.png", GRAY, "image/png")})
        c.post("/v1/predict", files={"file": ("b.jpg", PHOTO, "image/jpeg")})
        text = c.get("/metrics").text
    assert 'status="rejected_input",prediction="none"} 1' in text
    assert "neuroscan_request_latency_ms_count 2" in text
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["status"] for e in entries][1] == "rejected_input"
    assert all("image" not in key and "file" not in key for e in entries for key in e)
