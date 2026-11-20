"""Artifact-store protocol and local/GCS filesystem implementations."""

from __future__ import annotations

import io
from typing import Any

import joblib
import pandas as pd
from google.cloud import storage

from challenge.config import get_settings
from challenge.connectors import bigquery as bigquery_connector


class GCSArtifactStore:
    """Save and load model artifacts on Google Cloud Storage.

    The bucket name is read from ``Settings.gcp_bucket_artifacts`` at call time.

    Layout::

        gs://<bucket>/
            <run_id>/model.joblib
            latest.txt           ← contains the most-recent run_id
    """

    def _bucket_name(self) -> str:
        return get_settings().gcp_bucket_artifacts

    def save(self, model: Any, run_id: str) -> str:
        """Serialize and upload model to GCS; update latest.txt."""
        bucket_name = self._bucket_name()
        bucket = storage.Client().bucket(bucket_name)

        buffer = io.BytesIO()
        joblib.dump(model._model, buffer)
        model_key = f"{run_id}/model.joblib"
        bucket.blob(model_key).upload_from_string(buffer.getvalue())
        bucket.blob("latest.txt").upload_from_string(run_id)
        return f"gs://{bucket_name}/{model_key}"

    def load(self, model: Any, run_id: str | None) -> None:
        """Download and deserialize model from GCS into model._model."""
        bucket = storage.Client().bucket(self._bucket_name())
        if run_id is None:
            latest = bucket.blob("latest.txt")
            if not latest.exists():
                msg = "latest.txt not found in artifacts bucket; train a model first"
                raise FileNotFoundError(msg)
            run_id = latest.download_as_text().strip()
        payload = bucket.blob(f"{run_id}/model.joblib").download_as_bytes()
        model._model = joblib.load(io.BytesIO(payload))
        model._run_id = run_id


class GCSCSVClient:
    """Read flight data as a CSV object from the GCS input bucket.

    Predictions are written to BigQuery (not GCS) by delegating to an
    internal ``BigQueryClient`` — only the read side bypasses BigQuery.
    """

    def __init__(self, blob_name: str, _bq_writer: Any | None = None) -> None:
        """Initialise with the input-bucket object name and an optional injected writer."""
        self._blob_name = blob_name
        self._bq_writer = _bq_writer

    def read(self) -> pd.DataFrame:
        """Download and parse the CSV object from the configured input bucket."""
        bucket = storage.Client().bucket(get_settings().gcp_bucket_input)
        payload = bucket.blob(self._blob_name).download_as_bytes()
        return pd.read_csv(io.BytesIO(payload))

    def write(
        self,
        predictions: list[int],
        identifiers: pd.DataFrame,
        *,
        model_run_id: str | None = None,
    ) -> None:
        """Write predictions to the configured BigQuery predictions table."""
        writer = self._bq_writer or bigquery_connector.BigQueryClient()
        # Forward the run id only when known, so the bare two-argument call still works.
        extra = {} if model_run_id is None else {"model_run_id": model_run_id}
        writer.write(predictions, identifiers, **extra)
