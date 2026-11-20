"""Cloud Run Job entry point for model training."""

import logging
import sys
import uuid
from datetime import UTC, datetime

import pandas as pd

from challenge.config import get_settings
from challenge.connectors.bigquery import BigQueryClient
from challenge.connectors.local import LocalCSVClient
from challenge.model import DelayModel

logger = logging.getLogger(__name__)


TARGET_COLUMN = "delay"


def step_load_data(model: DelayModel, project: str, dataset: str) -> pd.DataFrame:
    """Load raw flight data through the model's injected reader.

    Parameters
    ----------
    model : DelayModel
        Model carrying the injected ``DataReader``.
    project : str
        GCP project id (unused by the local reader).
    dataset : str
        BigQuery dataset (unused by the local reader).

    Returns
    -------
    pd.DataFrame
        Raw flight records.
    """
    data = model.load_data(project, dataset)
    logger.info("Loaded %d raw rows", len(data))
    return data


def step_preprocess(
    model: DelayModel, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix and delay target from raw data."""
    features, target = model.preprocess(data, target_column=TARGET_COLUMN)
    logger.info(
        "Preprocessed features=%s, delayed share=%.3f",
        features.shape,
        float(target.iloc[:, 0].mean()),
    )
    return features, target


def step_train(model: DelayModel, features: pd.DataFrame, target: pd.DataFrame) -> None:
    """Fit the model on the preprocessed data."""
    model.fit(features, target)
    logger.info("Model fitted")


def step_save(model: DelayModel, bucket: str, run_id: str) -> str:
    """Persist the fitted model and return the artifact URI."""
    uri = model.save(bucket, run_id)
    logger.info("Saved model run_id=%s to %s", run_id, uri)
    return uri


def _new_run_id() -> str:
    """Return a sortable, unique run id (UTC timestamp + random suffix)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def main() -> str:
    """Orchestrate the full training pipeline and return the artifact URI."""
    settings = get_settings()
    if settings.deployment_mode == "gcp":
        from challenge.connectors.gcs import GCSArtifactStore  # noqa: PLC0415

        model = DelayModel(reader=BigQueryClient(), store=GCSArtifactStore())
    else:
        model = DelayModel(reader=LocalCSVClient("data.csv"))

    run_id = _new_run_id()
    logger.info("Starting training run %s (%s mode)", run_id, settings.deployment_mode)
    data = step_load_data(model, settings.gcp_project_id, settings.bq_dataset)
    features, target = step_preprocess(model, data)
    step_train(model, features, target)
    return step_save(model, settings.gcp_bucket_artifacts, run_id)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        main()
    except Exception:
        logger.exception("Training job failed")
        sys.exit(1)
