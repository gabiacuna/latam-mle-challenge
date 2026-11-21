"""Local end-to-end test: train a model, then run the serving pipeline on the real batch."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from challenge.config import clear_settings_cache
from challenge.serving import main as serving_main
from challenge.training import main as training_main

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COLUMNS = [
    "OPERA",
    "TIPOVUELO",
    "MES",
    "prediction",
    "model_run_id",
    "predicted_at",
]


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    """Point DATA_DIR/ARTIFACTS_DIR at tmp dirs seeded with the real CSVs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shutil.copy(REPO_ROOT / "data" / "data.csv", data_dir / "data.csv")
    shutil.copy(
        REPO_ROOT / "data" / "serving_input.csv", data_dir / "serving_input.csv"
    )
    artifacts_dir = tmp_path / "artifacts"

    monkeypatch.setenv("DEPLOYMENT_MODE", "local")
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))
    clear_settings_cache()
    yield data_dir, artifacts_dir
    clear_settings_cache()


def test_serving_e2e_local_writes_predictions_for_every_input_row(local_env):
    """Train, serve on serving_input.csv, and check predictions.csv end to end."""
    data_dir, artifacts_dir = local_env
    training_main()
    run_id = (artifacts_dir / "latest.txt").read_text().strip()

    n_predictions = serving_main()

    serving_input = pd.read_csv(data_dir / "serving_input.csv")
    output = pd.read_csv(artifacts_dir / "predictions.csv")
    assert n_predictions == len(serving_input)
    assert len(output) == len(serving_input)
    assert list(output.columns) == EXPECTED_COLUMNS
    assert set(output["prediction"].unique()) <= {0, 1}
    assert set(output["model_run_id"].unique()) == {run_id}
    assert output["predicted_at"].notna().all()
