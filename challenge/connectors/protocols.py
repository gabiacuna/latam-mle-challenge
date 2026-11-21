"""Protocols for data connectors"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd


class FlightIdentifiers:
    """Columns that identify a flight row in the predictions table."""

    COLUMNS: tuple[str, ...] = ("OPERA", "TIPOVUELO", "MES")

    @classmethod
    def select(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``data`` restricted to the identifier columns."""
        return data[list(cls.COLUMNS)].copy()


def build_predictions_frame(
    predictions: list[int],
    identifiers: pd.DataFrame,
    model_run_id: str | None = None,
) -> pd.DataFrame:
    """Build the predictions table rows shared by the local and BigQuery writers.

    Columns are the identifiers, then ``prediction``. When ``model_run_id`` is given,
    ``model_run_id`` and ``predicted_at`` (UTC) are appended.
    """
    output = FlightIdentifiers.select(identifiers).reset_index(drop=True)
    output["prediction"] = list(predictions)
    if model_run_id is not None:
        output["model_run_id"] = model_run_id
        output["predicted_at"] = datetime.now(UTC)
    return output


@runtime_checkable
class DataReader(Protocol):
    """Interface for loading raw flight data and writing predictions."""

    def read(self) -> pd.DataFrame:
        """Return a DataFrame containing raw flight records."""
        ...

    def write(
        self,
        predictions: list[int],
        identifiers: pd.DataFrame,
        *,
        model_run_id: str | None = None,
    ) -> None:
        """Persist predictions, paired with their OPERA/TIPOVUELO/MES identifiers."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Interface for persisting and retrieving trained model artifacts."""

    def save(self, model: Any, run_id: str) -> str:
        """Persist model to storage and return the artifact path."""
        ...

    def load(self, model: Any, run_id: str | None) -> None:
        """Deserialise artifact into model._model; resolve latest when run_id is None."""
        ...

    def latest_run_id(self) -> str | None:
        """Return the run id recorded as latest, or ``None`` when no model was saved."""
        ...

    def save_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        """Persist the evaluation metrics of a run next to its model."""
        ...

    def load_metrics(self, run_id: str) -> dict[str, Any]:
        """Return the metrics of a run, or an empty dict when none were saved."""
        ...
