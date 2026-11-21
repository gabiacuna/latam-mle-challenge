"""Cloud Run Job entry point for model training."""

import logging
import sys
import uuid
from datetime import UTC, datetime

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from challenge.config import get_settings
from challenge.connectors.bigquery import BigQueryClient
from challenge.connectors.local import LocalCSVClient
from challenge.model import DelayModel

logger = logging.getLogger(__name__)


TARGET_COLUMN = "delay"
HOLDOUT_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
MIN_CLASS_FOR_STRATIFY = 2


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


def holdout_metrics(
    y_true: pd.Series, y_pred: list[int], n_train: int, n_test: int
) -> dict[str, float | int]:
    """Return recall, precision, F1 of the delayed class plus accuracy on the holdout."""
    return {
        "recall_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_1": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "n_train": n_train,
        "n_test": n_test,
    }


def step_train(
    model: DelayModel, features: pd.DataFrame, target: pd.DataFrame
) -> dict[str, float | int]:
    """Fit the model on a stratified train split and return holdout metrics.

    The persisted model is the one fitted here (train split only), so the reported
    metrics describe the artifact that serving will load. Only one fit is performed.
    """
    labels = target.iloc[:, 0]
    stratify = labels if labels.value_counts().min() >= MIN_CLASS_FOR_STRATIFY else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=HOLDOUT_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=stratify,
    )
    model.fit(x_train, y_train.to_frame(TARGET_COLUMN))
    metrics = holdout_metrics(
        y_test, model.predict(x_test), n_train=len(x_train), n_test=len(x_test)
    )
    logger.info("Holdout metrics: %s", metrics)
    return metrics


def step_save(
    model: DelayModel,
    bucket: str,
    run_id: str,
    metrics: dict[str, float | int] | None = None,
) -> str:
    """Persist the fitted model and its metrics, and return the artifact URI."""
    uri = model.save(bucket, run_id)
    if metrics is not None:
        model.save_metrics(run_id, metrics)
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
    metrics = step_train(model, features, target)
    return step_save(model, settings.gcp_bucket_artifacts, run_id, metrics)


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
