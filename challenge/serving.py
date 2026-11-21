"""Cloud Run Job entry point for model serving."""

import logging
import sys

import pandas as pd

from challenge.config import get_settings
from challenge.connectors.local import LocalCSVClient
from challenge.connectors.protocols import DataReader, FlightIdentifiers
from challenge.model import DelayModel

logger = logging.getLogger(__name__)

SERVING_BLOB = "serving/serving_input.csv"

### Define here all the step for the pipeline implementation ###


def step_load_model(model: DelayModel) -> None:
    """Restore the latest persisted model through the injected artifact store.

    Parameters
    ----------
    model : DelayModel
        Model carrying the injected ``ArtifactStore``.
    """
    model.load()
    logger.info("Loaded model run_id=%s", model.run_id)


def step_load_data(model: DelayModel, project: str, dataset: str) -> pd.DataFrame:
    """Load the serving batch through the model's injected reader.

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
        Raw serving records.
    """
    data = model.load_data(project, dataset)
    logger.info("Loaded %d serving rows", len(data))
    return data


def step_preprocess(model: DelayModel, data: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for inference (no target).

    Parameters
    ----------
    model : DelayModel
        Model whose ``preprocess`` builds the training feature layout.
    data : pd.DataFrame
        Raw serving records.

    Returns
    -------
    pd.DataFrame
        Feature matrix with one row per input record.
    """
    features = model.preprocess(data, target_column=None)
    logger.info("Preprocessed features=%s", features.shape)
    return features


def step_predict(model: DelayModel, features: pd.DataFrame) -> list[int]:
    """Predict delay labels for each feature row.

    Parameters
    ----------
    model : DelayModel
        Model used for inference.
    features : pd.DataFrame
        Feature matrix from ``step_preprocess``.

    Returns
    -------
    list[int]
        0 (on time) or 1 (delayed) per row.
    """
    predictions = model.predict(features)
    delayed = sum(predictions) / len(predictions) if predictions else 0.0
    logger.info("Predicted %d rows, delayed share=%.3f", len(predictions), delayed)
    return predictions


def step_write(
    reader: DataReader,
    predictions: list[int],
    data: pd.DataFrame,
    model_run_id: str | None = None,
) -> None:
    """Persist predictions alongside their OPERA/TIPOVUELO/MES identifiers.

    Parameters
    ----------
    reader : DataReader
        Writer for the predictions destination (BigQuery or local CSV).
    predictions : list[int]
        Predictions, row-aligned with ``data``.
    data : pd.DataFrame
        Raw serving records; only the identifier columns are written.
    model_run_id : str, optional
        Run id of the model that produced the predictions.
    """
    reader.write(predictions, FlightIdentifiers.select(data), model_run_id=model_run_id)
    logger.info("Wrote %d predictions", len(predictions))


def main() -> int:
    """Orchestrate the full serving pipeline and return the number of predictions."""
    settings = get_settings()
    if settings.deployment_mode == "gcp":
        from challenge.connectors.gcs import (  # noqa: PLC0415
            GCSArtifactStore,
            GCSCSVClient,
        )

        reader = GCSCSVClient(blob_name=SERVING_BLOB)
        model = DelayModel(reader=reader, store=GCSArtifactStore())
    else:
        reader = LocalCSVClient("serving_input.csv")
        model = DelayModel(reader=reader)

    logger.info("Starting serving run (%s mode)", settings.deployment_mode)
    step_load_model(model)
    data = step_load_data(model, settings.gcp_project_id, settings.bq_dataset)
    features = step_preprocess(model, data)
    predictions = step_predict(model, features)
    step_write(reader, predictions, data, model_run_id=model.run_id)
    return len(predictions)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        main()
    except Exception:
        logger.exception("Serving job failed")
        sys.exit(1)
