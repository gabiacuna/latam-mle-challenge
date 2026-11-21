# ruff: noqa: PLR2004 - HTTP status codes and expected counts are literal in assertions
"""Local-mode end-to-end test of the pipeline control API (no GCP credentials needed)."""

import shutil
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from challenge.api.api import app
from challenge.api.predictions import get_repository
from challenge.config import clear_settings_cache

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Run the API in local mode against temporary data and artifact directories."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(REPO_ROOT / "data" / "data.csv", data_dir / "data.csv")
    shutil.copy(
        REPO_ROOT / "data" / "serving_input.csv", data_dir / "serving_input.csv"
    )
    monkeypatch.setenv("DEPLOYMENT_MODE", "local")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    clear_settings_cache()
    get_repository().invalidate()
    yield TestClient(app)
    get_repository().invalidate()
    clear_settings_cache()


def test_train_predict_results_end_to_end(client, tmp_path):
    """Train, serve, then query the stored predictions with a filter and pagination."""
    train = client.post("/pipeline/train")
    assert train.status_code == 200, train.text
    body = train.json()
    assert body["model_display_name"] == "delay-classifier"
    run_id = body["model_version"]
    assert run_id is not None
    assert body["metrics"]["recall_1"] is not None
    assert (tmp_path / "artifacts" / run_id / "metrics.json").is_file()

    predict = client.post("/pipeline/predict")
    assert predict.status_code == 200, predict.text
    predicted = predict.json()
    assert predicted["total_predictions"] > 0
    assert len(predicted["predictions"]) == min(10, predicted["total_predictions"])

    results = client.get("/pipeline/predict/results?mes=6&page=2&page_size=5")
    assert results.status_code == 200, results.text
    page = results.json()
    assert page["page"] == 2
    assert page["page_size"] == 5
    assert page["total_predictions"] > 0
    assert len(page["predictions"]) <= 5
    assert all(row["MES"] == 6 for row in page["predictions"])


def test_results_before_any_prediction_returns_empty_page(client):
    """A cold start with no predictions file answers 200 with an empty page."""
    response = client.get("/pipeline/predict/results")
    assert response.status_code == 200
    assert response.json() == {
        "total_predictions": 0,
        "page": 1,
        "page_size": 10,
        "predictions": [],
    }


def test_concurrent_training_returns_409_for_second_request(client):
    """While one training run holds the local lock, a second request gets 409."""
    started = threading.Event()
    release = threading.Event()

    def _slow_training() -> str:
        started.set()
        release.wait(timeout=30)
        return "artifacts/run/model.joblib"

    first: dict = {}

    def _first_request() -> None:
        first["response"] = client.post("/pipeline/train")

    with patch("challenge.training.main", side_effect=_slow_training):
        worker = threading.Thread(target=_first_request)
        worker.start()
        try:
            assert started.wait(timeout=30)
            second = client.post("/pipeline/train")
        finally:
            release.set()
            worker.join(timeout=60)

    assert second.status_code == 409
    assert "already running" in second.json()["detail"]
    assert first["response"].status_code == 200
