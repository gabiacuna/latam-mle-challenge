# ruff: noqa: PLR2004, D103 - literal HTTP codes in assertions; test names describe the contract
"""Contract tests: every endpoint is documented in OpenAPI and answers its documented codes."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from google.api_core.exceptions import ServiceUnavailable

os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_REGION", "us-central1")
os.environ.setdefault("GCP_BUCKET_ARTIFACTS", "test-project-mle-artifacts")
os.environ.setdefault("BQ_DATASET", "mle_challenge")

from challenge.api.api import app
from challenge.api.jobs import PipelineBusyError
from challenge.api.schemas import (
    ModelMetrics,
    PredictionRow,
    PredictResponse,
    TrainResponse,
)

ENDPOINTS = {
    ("get", "/health"): {"tag": "health", "codes": {"200"}},
    ("post", "/pipeline/train"): {
        "tag": "pipeline",
        "codes": {"200", "409", "422", "500", "503"},
    },
    ("post", "/pipeline/predict"): {
        "tag": "pipeline",
        "codes": {"200", "409", "422", "500", "503"},
    },
    ("get", "/pipeline/predict/results"): {
        "tag": "pipeline",
        "codes": {"200", "422", "503"},
    },
}


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def spec(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def _unavailable():
    return ServiceUnavailable("bigquery down")


# --- documentation ---


def test_openapi_lists_every_endpoint_and_method(spec):
    for method, path in ENDPOINTS:
        assert method in spec["paths"][path], (
            f"{method.upper()} {path} missing from OpenAPI"
        )


def test_openapi_metadata_is_set(spec):
    assert spec["info"]["title"] == "LATAM MLE Flight Delay API"
    assert spec["info"]["version"] == "1.0.0"
    assert spec["info"]["description"].strip()
    assert {tag["name"] for tag in spec["tags"]} == {"health", "pipeline"}


@pytest.mark.parametrize(("method", "path"), list(ENDPOINTS))
def test_operation_is_fully_documented(spec, method, path):
    spec_op = spec["paths"][path][method]
    expected = ENDPOINTS[(method, path)]
    assert spec_op["tags"] == [expected["tag"]]
    assert spec_op["summary"].strip()
    assert spec_op["description"].strip()
    assert set(spec_op["responses"]) == expected["codes"]
    for code, response in spec_op["responses"].items():
        assert response["description"].strip(), f"{path} {code} has no description"


@pytest.mark.parametrize(("method", "path"), [k for k in ENDPOINTS if k[0] == "post"])
def test_error_responses_reference_error_schema(spec, method, path):
    responses = spec["paths"][path][method]["responses"]
    for code, response in responses.items():
        if code == "200":
            continue
        content = response.get("content", {}).get("application/json", {})
        assert "schema" in content, f"{path} {code} has no schema"


def test_query_parameters_are_documented(spec):
    params = {
        p["name"]: p
        for p in spec["paths"]["/pipeline/predict/results"]["get"]["parameters"]
    }
    assert set(params) == {"page", "page_size", "opera", "tipovuelo", "mes"}
    for name, param in params.items():
        assert param.get("description"), f"query parameter {name} has no description"
    assert params["page_size"]["schema"]["maximum"] == 100
    assert params["page"]["schema"]["minimum"] == 1


def test_prediction_fields_are_documented(spec):
    row = spec["components"]["schemas"]["PredictionRow"]["properties"]
    for field in ("OPERA", "TIPOVUELO", "MES", "predicted_delay"):
        assert row[field].get("description"), (
            f"PredictionRow.{field} has no description"
        )


def test_examples_validate_against_response_models():
    """Examples shipped in the docs must be valid instances of their models."""

    metrics = ModelMetrics.model_validate(
        ModelMetrics.model_config["json_schema_extra"]["example"]
    )
    TrainResponse(
        model_version="20260101T120000",
        pipeline_job_id="projects/p/locations/us-central1/jobs/training/executions/abc",
        metrics=metrics,
    )
    PredictionRow.model_validate(
        PredictionRow.model_config["json_schema_extra"]["example"]
    )
    PredictResponse(
        total_predictions=1,
        pipeline_job_id="x",
        predictions=[PredictionRow.model_config["json_schema_extra"]["example"]],
    )


# --- GET /health ---


def test_health_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- POST /pipeline/train ---


def test_train_422_when_raw_flights_empty(client):
    with patch("challenge.api.api._check_raw_flights", return_value=0):
        response = client.post("/pipeline/train")
    assert response.status_code == 422
    assert response.json()["detail"] == "raw_flights table is empty"


def test_train_409_when_job_already_running(client):
    with (
        patch("challenge.api.api._check_raw_flights", return_value=10),
        patch(
            "challenge.api.api._submit_and_wait",
            side_effect=PipelineBusyError("job is already running"),
        ),
    ):
        response = client.post("/pipeline/train")
    assert response.status_code == 409
    assert "already running" in response.json()["detail"]


def test_train_500_on_pipeline_failure(client):
    with (
        patch("challenge.api.api._check_raw_flights", return_value=10),
        patch("challenge.api.api._submit_and_wait", side_effect=RuntimeError("boom")),
    ):
        response = client.post("/pipeline/train")
    assert response.status_code == 500
    body = response.json()["detail"]
    assert body["message"] == "training pipeline failed"
    assert body["details"] == "boom"


def test_train_503_when_bigquery_unavailable(client):
    with patch("challenge.api.api._check_raw_flights", side_effect=_unavailable()):
        response = client.post("/pipeline/train")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


# --- POST /pipeline/predict ---


def test_predict_409_when_job_already_running(client):
    with (
        patch("challenge.api.api._check_model_exists", return_value=True),
        patch(
            "challenge.api.api._submit_and_wait",
            side_effect=PipelineBusyError("job is already running"),
        ),
    ):
        response = client.post("/pipeline/predict")
    assert response.status_code == 409


def test_predict_503_when_storage_unavailable(client):
    with patch("challenge.api.api._check_model_exists", side_effect=_unavailable()):
        response = client.post("/pipeline/predict")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def test_predict_200_returns_preview_capped_at_ten(client, mock_job):
    rows = [
        {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 1, "predicted_delay": 0}
    ] * 10
    with (
        patch("challenge.api.api._check_model_exists", return_value=True),
        patch("challenge.api.api._submit_and_wait", return_value=mock_job),
        patch(
            "challenge.api.api._query_predictions",
            return_value={"total_predictions": 500, "predictions": rows},
        ) as query,
    ):
        response = client.post("/pipeline/predict")
    assert response.status_code == 200
    assert query.call_args[0][:2] == (1, 10)  # first page, preview size 10
    assert len(response.json()["predictions"]) == 10


@pytest.fixture
def mock_job():
    class _Job:
        name = "projects/test/locations/us-central1/jobs/serving/executions/1"

    return _Job()


# --- GET /pipeline/predict/results ---


@pytest.mark.parametrize(
    "query",
    ["page=0", "page_size=0", "page_size=101", "page=abc"],
)
def test_results_422_on_out_of_range_pagination(client, query):
    with patch("challenge.api.api._query_predictions") as query_mock:
        response = client.get(f"/pipeline/predict/results?{query}")
    assert response.status_code == 422
    query_mock.assert_not_called()


def test_results_422_on_invalid_mes(client):
    response = client.get("/pipeline/predict/results?mes=june")
    assert response.status_code == 422
    assert response.json()["detail"] == "mes must be a comma-separated list of integers"


def test_results_503_when_bigquery_unavailable(client):
    with patch("challenge.api.api._query_predictions", side_effect=_unavailable()):
        response = client.get("/pipeline/predict/results")
    assert response.status_code == 503


def test_results_response_shape(client):
    rows = [{"OPERA": "Sky Airline", "TIPOVUELO": "N", "MES": 3, "predicted_delay": 1}]
    with patch(
        "challenge.api.api._query_predictions",
        return_value={"total_predictions": 1, "predictions": rows},
    ):
        response = client.get("/pipeline/predict/results?page=1&page_size=10")
    assert response.status_code == 200
    assert response.json() == {
        "total_predictions": 1,
        "page": 1,
        "page_size": 10,
        "predictions": rows,
    }
