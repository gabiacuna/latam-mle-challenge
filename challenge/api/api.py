"""Pipeline control FastAPI — triggers Cloud Run Job training/serving pipelines.

Routes only validate input, call a collaborator and map failures to HTTP status codes.
Pipeline execution lives in ``jobs.py`` and the predictions query in ``predictions.py``.
Routes that wait on jobs or BigQuery are plain ``def`` so FastAPI runs them in its
threadpool instead of blocking the event loop.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from google.api_core.exceptions import GoogleAPICallError

from challenge.api.checks import latest_model_metadata, model_exists, raw_flight_count
from challenge.api.jobs import PipelineBusyError, PipelineJob, get_runner
from challenge.api.predictions import get_repository
from challenge.api.schemas import (
    ModelMetrics,
    PredictionRow,
    PredictResponse,
    ResultsResponse,
    TrainResponse,
)
from challenge.api.utils import parse_csv_list, parse_mes
from challenge.config import get_settings

logger = logging.getLogger(__name__)

PREDICT_PREVIEW_SIZE = 10
_REQUIRED_JOB_SETTINGS = {
    "CLOUD_RUN_TRAINING_JOB": "cloud_run_training_job",
    "CLOUD_RUN_SERVING_JOB": "cloud_run_serving_job",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail fast in gcp mode when the Cloud Run job names are not configured."""
    settings = get_settings()
    if settings.deployment_mode == "gcp":
        missing = [
            env
            for env, field in _REQUIRED_JOB_SETTINGS.items()
            if not getattr(settings, field)
        ]
        if missing:
            msg = f"Missing required job settings in gcp mode: {', '.join(missing)}"
            raise RuntimeError(msg)
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(GoogleAPICallError)
async def _storage_unavailable(
    request: Request, exc: GoogleAPICallError
) -> JSONResponse:
    """Map BigQuery/GCS/Cloud Run API errors to 503 with a generic body."""
    logger.error("Upstream call failed on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "BigQuery or GCS is unavailable; try again later"},
    )


# --- collaborators (patched by tests; each is a one-line delegation) ---


def _check_raw_flights() -> int:
    return raw_flight_count()


def _check_model_exists() -> bool:
    return model_exists()


def _submit_and_wait(job: PipelineJob) -> Any:
    return get_runner().submit_and_wait(job)


def _query_predictions(
    page: int,
    page_size: int,
    opera: list[str] | None,
    tipovuelo: list[str] | None,
    mes: list[int] | None,
) -> dict[str, Any]:
    return get_repository().query(page, page_size, opera, tipovuelo, mes)


def _run_pipeline(job: PipelineJob) -> Any:
    """Run a pipeline job and map its failures to 409 (busy) or 500 (failed)."""
    try:
        execution = _submit_and_wait(job)
    except PipelineBusyError as exc:
        logger.warning("%s pipeline rejected: %s", job, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("%s pipeline failed", job)
        raise HTTPException(
            status_code=500,
            detail={"message": f"{job} pipeline failed", "details": str(exc)},
        ) from exc
    logger.info("%s pipeline finished execution=%s", job, execution.name)
    return execution


# --- routes ---


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Return liveness status."""
    return {"status": "ok"}


@app.post("/pipeline/train", status_code=200, response_model=TrainResponse)
def post_pipeline_train() -> TrainResponse:
    """Trigger the training Cloud Run Job and return model metadata."""
    if _check_raw_flights() == 0:
        raise HTTPException(status_code=422, detail="raw_flights table is empty")
    execution = _run_pipeline("training")
    run_id, metrics = latest_model_metadata()
    logger.info("train finished execution=%s run_id=%s", execution.name, run_id)
    return TrainResponse(
        model_version=run_id,
        pipeline_job_id=execution.name,
        metrics=ModelMetrics.model_validate(metrics),
    )


@app.post("/pipeline/predict", status_code=200, response_model=PredictResponse)
def post_pipeline_predict() -> PredictResponse:
    """Trigger the serving Cloud Run Job and return first 10 predictions."""
    if not _check_model_exists():
        raise HTTPException(
            status_code=422, detail="no trained model found; run /pipeline/train first"
        )
    execution = _run_pipeline("serving")
    get_repository().invalidate()
    result = _query_predictions(1, PREDICT_PREVIEW_SIZE, None, None, None)
    logger.info("predict finished execution=%s", execution.name)
    return PredictResponse(
        total_predictions=result["total_predictions"],
        pipeline_job_id=execution.name,
        predictions=[PredictionRow(**row) for row in result["predictions"]],
    )


@app.get("/pipeline/predict/results", status_code=200, response_model=ResultsResponse)
def get_prediction_results(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    opera: str | None = None,
    tipovuelo: str | None = None,
    mes: str | None = None,
) -> ResultsResponse:
    """Return a paginated page of predictions, optionally filtered by OPERA/TIPOVUELO/MES."""
    try:
        mes_values = parse_mes(mes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = _query_predictions(
        page, page_size, parse_csv_list(opera), parse_csv_list(tipovuelo), mes_values
    )
    logger.info(
        "results page=%s page_size=%s total=%s",
        page,
        page_size,
        result["total_predictions"],
    )
    return ResultsResponse(
        total_predictions=result["total_predictions"],
        page=page,
        page_size=page_size,
        predictions=[PredictionRow(**row) for row in result["predictions"]],
    )
