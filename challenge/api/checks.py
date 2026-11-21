"""Pre-flight checks and the artifact store used by the API."""

from __future__ import annotations

import logging
from functools import cache
from typing import Any

from challenge.config import get_settings
from challenge.connectors.protocols import ArtifactStore

logger = logging.getLogger(__name__)


@cache
def _build_store(mode: str) -> ArtifactStore:
    if mode == "gcp":
        from challenge.connectors.gcs import GCSArtifactStore  # noqa: PLC0415

        return GCSArtifactStore()
    from challenge.connectors.local import LocalArtifactStore  # noqa: PLC0415

    return LocalArtifactStore()


@cache
def _bigquery_client() -> Any:
    from google.cloud import bigquery  # noqa: PLC0415

    return bigquery.Client(project=get_settings().gcp_project_id)


def get_artifact_store() -> ArtifactStore:
    """Return the artifact store for the configured deployment mode."""
    return _build_store(get_settings().deployment_mode)


def raw_flight_count() -> int:
    """Return the number of raw flight rows (BigQuery in gcp mode, data/data.csv locally)."""
    if get_settings().deployment_mode == "gcp":
        settings = get_settings()
        table = (
            f"{settings.gcp_project_id}.{settings.bq_dataset}.{settings.bq_raw_table}"
        )
        query = f"SELECT COUNT(*) AS n FROM `{table}`"
        rows = _bigquery_client().query(query).result()
        return int(next(iter(rows))["n"])

    from challenge.connectors.local import LocalCSVClient  # noqa: PLC0415

    return len(LocalCSVClient("data.csv").read())


def model_exists() -> bool:
    """Return whether at least one trained model artifact exists."""
    return get_artifact_store().latest_run_id() is not None


def latest_model_metadata() -> tuple[str | None, dict]:
    """Return ``(run_id, metrics)`` of the latest artifact, degrading to ``(None, {})``.

    Used after training: the run has already succeeded, so a failed metadata read is
    logged and reported as missing rather than failing the request.
    """
    try:
        store = get_artifact_store()
        run_id = store.latest_run_id()
        metrics = store.load_metrics(run_id) if run_id is not None else {}
    except Exception:
        logger.warning("Could not read model metadata", exc_info=True)
        return None, {}
    return run_id, metrics
