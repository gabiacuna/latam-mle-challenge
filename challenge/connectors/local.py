"""Data-reader and Artifactor local implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from challenge.connectors.protocols import build_predictions_frame


class LocalCSVClient:
    """Read a CSV from the local data directory and write predictions to disk.

    The data directory is taken from the ``DATA_DIR`` environment variable,
    defaulting to ``data/`` relative to the current working directory.

    When ``filename`` is ``None`` (default): globs for the single ``.csv`` in
    the directory, raising ``ValueError`` on multiples and ``FileNotFoundError``
    on none — preserving original training behaviour.

    When ``filename`` is provided: reads that specific file directly, raising
    ``FileNotFoundError`` if it does not exist.
    """

    def __init__(self, filename: str | None = None) -> None:
        """Initialise the reader with an optional specific filename."""
        self._filename = filename

    def _data_dir(self) -> Path:
        return Path(os.environ.get("DATA_DIR", "data"))

    def read(self) -> pd.DataFrame:
        """Return a DataFrame from the target CSV file."""
        if self._filename is not None:
            path = self._data_dir() / self._filename
            if not path.is_file():
                msg = f"CSV file not found: {path}"
                raise FileNotFoundError(msg)
        else:
            csv_files = sorted(self._data_dir().glob("*.csv"))
            if not csv_files:
                msg = f"No CSV files found in {self._data_dir()}"
                raise FileNotFoundError(msg)
            if len(csv_files) > 1:
                msg = f"Expected a single CSV in {self._data_dir()}, found {len(csv_files)}"
                raise ValueError(msg)
            path = csv_files[0]
        return pd.read_csv(path)

    def write(
        self,
        predictions: list[int],
        identifiers: pd.DataFrame,
        *,
        model_run_id: str | None = None,
    ) -> None:
        """Write predictions to ``<ARTIFACTS_DIR>/predictions.csv``.

        Creates the artifacts directory if it does not exist.
        Output columns: ``OPERA``, ``TIPOVUELO``, ``MES`` (row-aligned with
        predictions) followed by ``prediction`` (0|1). When ``model_run_id`` is
        given, ``model_run_id`` and ``predicted_at`` (UTC ISO timestamp) are
        appended, mirroring the BigQuery predictions table.
        """
        root = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
        root.mkdir(parents=True, exist_ok=True)
        build_predictions_frame(predictions, identifiers, model_run_id).to_csv(
            root / "predictions.csv", index=False
        )


class LocalArtifactStore:
    """Save and load model artifacts on the local filesystem.

    The root directory is taken from the ``ARTIFACTS_DIR`` environment variable,
    defaulting to ``artifacts/`` relative to the current working directory.

    Layout::

        <ARTIFACTS_DIR>/
            <run_id>/model.joblib
            latest.txt           ← contains the most-recent run_id
    """

    def _root(self) -> Path:
        return Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))

    def save(self, model: Any, run_id: str) -> str:
        """Save model to <ARTIFACTS_DIR>/<run_id>/model.joblib and update latest.txt."""
        run_dir = self._root() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = run_dir / "model.joblib"
        joblib.dump(model._model, artifact_path)
        (self._root() / "latest.txt").write_text(run_id)
        return str(artifact_path)

    def latest_run_id(self) -> str | None:
        """Return the run id stored in latest.txt, or ``None`` when it is absent."""
        latest = self._root() / "latest.txt"
        if not latest.is_file():
            return None
        return latest.read_text().strip()

    def save_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        """Write metrics to <ARTIFACTS_DIR>/<run_id>/metrics.json."""
        run_dir = self._root() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def load_metrics(self, run_id: str) -> dict[str, Any]:
        """Read <ARTIFACTS_DIR>/<run_id>/metrics.json, or ``{}`` when it is missing."""
        path = self._root() / run_id / "metrics.json"
        if not path.is_file():
            return {}
        return json.loads(path.read_text())

    def load(self, model: Any, run_id: str | None) -> None:
        """Load model from <ARTIFACTS_DIR>/<run_id>/model.joblib into model._model."""
        if run_id is None:
            run_id = self.latest_run_id()
            if run_id is None:
                msg = f"No latest.txt found in {self._root()}; train a model first"
                raise FileNotFoundError(msg)
        model._model = joblib.load(self._root() / run_id / "model.joblib")
        model._run_id = run_id
