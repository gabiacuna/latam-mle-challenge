"""Pydantic response models for the pipeline control API."""

from pydantic import BaseModel, Field

MODEL_DISPLAY_NAME = "delay-classifier"


class ModelMetrics(BaseModel):
    """Holdout metrics of the delayed class (class 1) written by the training job."""

    recall_1: float | None = None
    precision_1: float | None = None
    f1_1: float | None = None
    accuracy: float | None = None
    n_train: int | None = None
    n_test: int | None = None


class TrainResponse(BaseModel):
    """Outcome of a training run."""

    model_display_name: str = MODEL_DISPLAY_NAME
    model_version: str | None = Field(None, description="Run id of the trained model")
    pipeline_job_id: str
    metrics: ModelMetrics


class PredictionRow(BaseModel):
    """One predicted flight row."""

    OPERA: str
    TIPOVUELO: str
    MES: int
    predicted_delay: int


class PredictResponse(BaseModel):
    """Outcome of a serving run plus the first page of its predictions."""

    total_predictions: int
    pipeline_job_id: str
    predictions: list[PredictionRow]


class ResultsResponse(BaseModel):
    """One page of stored predictions."""

    total_predictions: int
    page: int
    page_size: int
    predictions: list[PredictionRow]
