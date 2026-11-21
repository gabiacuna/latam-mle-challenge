"""Pydantic response models for the pipeline control API."""

from pydantic import BaseModel, ConfigDict, Field

MODEL_DISPLAY_NAME = "delay-classifier"


class HealthResponse(BaseModel):
    """Liveness status of the API."""

    status: str = Field(
        description="Always `ok` while the process is serving requests."
    )


class ErrorResponse(BaseModel):
    """Error body returned for 4xx and 503 responses."""

    detail: str = Field(description="Human-readable error message.")


class PipelineFailureDetail(BaseModel):
    """Detail object returned when a pipeline job fails (HTTP 500)."""

    message: str = Field(
        description="Which pipeline failed, e.g. 'training pipeline failed'."
    )
    details: str = Field(description="Exception text raised by the job runner.")


class PipelineFailureResponse(BaseModel):
    """Error body returned when a pipeline job fails (HTTP 500)."""

    detail: PipelineFailureDetail


class ModelMetrics(BaseModel):
    """Holdout metrics of the delayed class (class 1) written by the training job."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recall_1": 0.62,
                "precision_1": 0.31,
                "f1_1": 0.41,
                "accuracy": 0.66,
                "n_train": 55000,
                "n_test": 13800,
            }
        }
    )

    recall_1: float | None = Field(
        None, description="Recall for the delayed class (1). Primary business metric."
    )
    precision_1: float | None = Field(
        None, description="Precision for the delayed class (1)."
    )
    f1_1: float | None = Field(None, description="F1 score for the delayed class (1).")
    accuracy: float | None = Field(
        None, description="Overall accuracy on the holdout set."
    )
    n_train: int | None = Field(None, description="Number of rows used for training.")
    n_test: int | None = Field(None, description="Number of rows in the holdout set.")


class TrainResponse(BaseModel):
    """Outcome of a training run."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_display_name": "delay-classifier",
                "model_version": "20260101T120000",
                "pipeline_job_id": "projects/p/locations/us-central1/jobs/training/executions/abc",
                "metrics": ModelMetrics.model_config["json_schema_extra"]["example"],
            }
        }
    )

    model_display_name: str = Field(
        MODEL_DISPLAY_NAME, description="Logical name of the model."
    )
    model_version: str | None = Field(None, description="Run id of the trained model.")
    pipeline_job_id: str = Field(
        description="Cloud Run execution name of the training job."
    )
    metrics: ModelMetrics = Field(
        description="Holdout metrics of the newly trained model."
    )


class PredictionRow(BaseModel):
    """One predicted flight row."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "OPERA": "Grupo LATAM",
                "TIPOVUELO": "I",
                "MES": 6,
                "predicted_delay": 0,
            }
        }
    )

    OPERA: str = Field(description="Airline operator name.")
    TIPOVUELO: str = Field(
        description="Flight type: 'I' (international) or 'N' (national)."
    )
    MES: int = Field(description="Month of the flight (1-12).")
    predicted_delay: int = Field(
        description="Predicted label: 1 = delayed, 0 = on time."
    )


class PredictResponse(BaseModel):
    """Outcome of a serving run plus the first page of its predictions."""

    total_predictions: int = Field(description="Total number of predictions stored.")
    pipeline_job_id: str = Field(
        description="Cloud Run execution name of the serving job."
    )
    predictions: list[PredictionRow] = Field(
        description="First 10 predictions produced by this run."
    )


class ResultsResponse(BaseModel):
    """One page of stored predictions."""

    total_predictions: int = Field(
        description="Total number of predictions matching the filters (not just this page)."
    )
    page: int = Field(description="1-based page number that was returned.")
    page_size: int = Field(description="Number of rows requested per page.")
    predictions: list[PredictionRow] = Field(description="Rows on this page.")
