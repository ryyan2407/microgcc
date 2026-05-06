from pathlib import Path

from fastapi.testclient import TestClient

from microgcc.api import create_app
from microgcc.training import train_all


def test_fast_training_and_api(tmp_path: Path):
    registry = train_all("data.csv", tmp_path, fast=True)
    assert registry["dataset_profile"]["state_count"] == 43
    assert registry["forecast_horizon_weeks"] == 8
    assert registry["mandatory_models"] == ["sarima", "prophet", "xgboost", "lstm"]
    assert "seasonal_naive" in registry["successful_models"]

    client = TestClient(create_app(tmp_path))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    prediction = client.post("/predict", json={"state": "California", "horizon": 3, "model": "best"})
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["count"] == 3
    assert body["predictions"][0]["state"] == "California"

    missing = client.post("/predict", json={"state": "Atlantis", "horizon": 1, "model": "best"})
    assert missing.status_code == 404
