"""Data-reader and Artifactor local implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

_IDENTIFIER_COLS = ["OPERA", "TIPOVUELO", "MES"]


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

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Write predictions to ``<ARTIFACTS_DIR>/predictions.csv``.

        Creates the artifacts directory if it does not exist.
        Output columns: ``OPERA``, ``TIPOVUELO``, ``MES`` (row-aligned with
        predictions) followed by ``prediction`` (0|1).
        """
        root = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
        root.mkdir(parents=True, exist_ok=True)
        output = identifiers[_IDENTIFIER_COLS].reset_index(drop=True)
        output["prediction"] = list(predictions)
        output.to_csv(root / "predictions.csv", index=False)


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

    def load(self, model: Any, run_id: str | None) -> None:
        """Load model from <ARTIFACTS_DIR>/<run_id>/model.joblib into model._model."""
        if run_id is None:
            latest = self._root() / "latest.txt"
            if not latest.is_file():
                msg = f"No latest.txt found in {self._root()}; train a model first"
                raise FileNotFoundError(msg)
            run_id = latest.read_text().strip()
        model._model = joblib.load(self._root() / run_id / "model.joblib")
