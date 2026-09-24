"""Integration tests for the FastAPI endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] == "Credit Risk API is live"


def test_health_ok_with_model(model_dir):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info_exposes_training_metadata(model_dir):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "test-1.0"
    assert "roc_auc" in body["metrics"]
    assert body["risk_threshold"] == 0.20
    assert body["n_features"] == 8


def test_predict_valid_application(model_dir, sample_application):
    resp = client.post("/predict", json=sample_application)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_level"] in ("High", "Low")
    assert body["threshold"] == 0.20


def test_predict_uses_calibrated_threshold_from_metrics(model_dir, sample_application):
    # Artifacts trained with threshold calibration carry the serving threshold
    # in metrics.json; the API must serve it instead of the fallback constant.
    import json

    import src.model as model_module

    data = json.loads(model_module.METRICS_PATH.read_text())
    data["metrics"]["threshold"] = 0.31
    model_module.METRICS_PATH.write_text(json.dumps(data))
    model_module.load_model_bundle.cache_clear()
    try:
        resp = client.post("/predict", json=sample_application)
        assert resp.status_code == 200
        assert resp.json()["threshold"] == 0.31
    finally:
        model_module.load_model_bundle.cache_clear()


def test_predict_rejects_unknown_gender(model_dir, sample_application):
    sample_application["CODE_GENDER"] = "X"
    resp = client.post("/predict", json=sample_application)
    assert resp.status_code == 422


def test_predict_rejects_negative_income(model_dir, sample_application):
    sample_application["AMT_INCOME_TOTAL"] = -1
    resp = client.post("/predict", json=sample_application)
    assert resp.status_code == 422


def test_predict_rejects_missing_fields(model_dir, sample_application):
    del sample_application["EXT_SOURCE_3"]
    resp = client.post("/predict", json=sample_application)
    assert resp.status_code == 422


def test_health_reports_missing_model(tmp_path, monkeypatch):
    import src.model as model_module

    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(model_module, "MODEL_COLUMNS_PATH", tmp_path / "missing_cols.pkl")
    model_module.load_model_bundle.cache_clear()
    try:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "model_not_loaded"}
    finally:
        model_module.load_model_bundle.cache_clear()
