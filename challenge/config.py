"""Centralised, lazily-loaded configuration for the challenge package."""

import os
from dataclasses import dataclass
from functools import lru_cache

_GCP_REQUIRED_FIELDS: dict[str, str] = {
    "gcp_project_id": "GCP_PROJECT_ID",
    "gcp_bucket_artifacts": "GCP_BUCKET_ARTIFACTS",
    "gcp_bucket_input": "GCP_BUCKET_INPUT",
    "bq_dataset": "BQ_DATASET",
    "bq_raw_table": "BQ_RAW_TABLE",
    "bq_predictions_table": "BQ_PREDICTIONS_TABLE",
}


@dataclass(frozen=True)
class Settings:
    """All environment-variable configuration in one place."""

    deployment_mode: str
    data_dir: str
    artifacts_dir: str
    gcp_project_id: str
    gcp_region: str
    gcp_bucket_artifacts: str
    gcp_bucket_input: str
    bq_dataset: str
    bq_raw_table: str
    bq_predictions_table: str

    def __post_init__(self) -> None:
        """Validate required GCP vars when deployment_mode is 'gcp'."""
        if self.deployment_mode not in {"local", "gcp"}:
            msg = f"DEPLOYMENT_MODE must be 'local' or 'gcp', got {self.deployment_mode!r}"
            raise ValueError(msg)
        if self.deployment_mode != "gcp":
            return
        missing = [
            env_name
            for field_name, env_name in _GCP_REQUIRED_FIELDS.items()
            if not getattr(self, field_name)
        ]
        if missing:
            msg = f"Missing required GCP settings in gcp mode: {', '.join(missing)}"
            raise ValueError(msg)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings read from environment variables."""
    return Settings(
        deployment_mode=os.environ.get("DEPLOYMENT_MODE", "local"),
        data_dir=os.environ.get("DATA_DIR", "data"),
        artifacts_dir=os.environ.get("ARTIFACTS_DIR", "artifacts"),
        gcp_project_id=os.environ.get("GCP_PROJECT_ID", ""),
        gcp_region=os.environ.get("GCP_REGION", "us-central1"),
        gcp_bucket_artifacts=os.environ.get("GCP_BUCKET_ARTIFACTS", ""),
        gcp_bucket_input=os.environ.get("GCP_BUCKET_INPUT", ""),
        bq_dataset=os.environ.get("BQ_DATASET", ""),
        bq_raw_table=os.environ.get("BQ_RAW_TABLE", ""),
        bq_predictions_table=os.environ.get("BQ_PREDICTIONS_TABLE", ""),
    )


def clear_settings_cache() -> None:
    """Invalidate the cached settings — call in test teardown after patching env vars."""
    get_settings.cache_clear()
