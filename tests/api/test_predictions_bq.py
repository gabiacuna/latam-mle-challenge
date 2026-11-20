# ruff: noqa: PLR2004 - HTTP status codes and expected counts are literal in assertions
"""BigQuery predictions query: parameterised SQL and prediction -> predicted_delay mapping."""

from google.cloud import bigquery

from challenge.api.predictions import BigQueryPredictionsRepository
from challenge.config import clear_settings_cache


class _FakeJob:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def result(self) -> list[dict]:
        return self._rows


class _FakeClient:
    def __init__(self, responses: list[list[dict]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, bigquery.QueryJobConfig]] = []

    def query(self, sql: str, job_config: bigquery.QueryJobConfig) -> _FakeJob:
        self.calls.append((sql, job_config))
        return _FakeJob(self._responses.pop(0))


def test_bigquery_query_uses_parameters_and_maps_prediction(monkeypatch):
    """User filters never reach the SQL text and the result maps prediction to predicted_delay."""
    monkeypatch.setenv("DEPLOYMENT_MODE", "gcp")
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")
    monkeypatch.setenv("GCP_BUCKET_ARTIFACTS", "test-artifacts")
    monkeypatch.setenv("GCP_BUCKET_INPUT", "test-input")
    monkeypatch.setenv("BQ_DATASET", "flights")
    monkeypatch.setenv("BQ_RAW_TABLE", "raw_flights")
    monkeypatch.setenv("BQ_PREDICTIONS_TABLE", "predictions")
    clear_settings_cache()

    injected = "x') OR 1=1 --"
    client = _FakeClient(
        responses=[
            [{"n": 1}],
            [{"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 6, "prediction": 1}],
        ]
    )
    repo = BigQueryPredictionsRepository(client)

    try:
        result = repo.query(
            page=1, page_size=10, opera=[injected], tipovuelo=None, mes=[6]
        )
    finally:
        clear_settings_cache()

    assert result == {
        "total_predictions": 1,
        "predictions": [
            {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 6, "predicted_delay": 1}
        ],
    }
    for sql, _config in client.calls:
        assert injected not in sql
        assert "UNNEST(@opera)" in sql
        assert "UNNEST(@mes)" in sql
        assert "TIPOVUELO IN" not in sql
    params = {p.name: p for p in client.calls[-1][1].query_parameters}
    assert params["opera"].values == [injected]
    assert params["mes"].values == [6]
    assert params["limit"].value == 10
    assert params["offset"].value == 0
