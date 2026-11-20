"""Pipeline runners: run the training/serving jobs on Cloud Run or in process."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import uuid
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, Protocol

from challenge.config import get_settings

logger = logging.getLogger(__name__)

PipelineJob = Literal["training", "serving"]

# Cloud Run job task timeout (config/base.yaml cloud_run.timeout) plus a margin for
# scheduling. The API service request timeout is set above this (900 s).
JOB_TIMEOUT_SECONDS = 600
WAIT_MARGIN_SECONDS = 60


class PipelineError(RuntimeError):
    """A pipeline job failed or did not finish in time."""


class PipelineBusyError(RuntimeError):
    """A pipeline job of the same kind is already running."""


@dataclass(frozen=True)
class PipelineExecution:
    """Minimal execution handle returned by the local runner."""

    name: str


class PipelineRunner(Protocol):
    """Runs one pipeline job to completion and returns its execution handle."""

    def submit_and_wait(self, job: PipelineJob) -> Any:
        """Run ``job`` synchronously; the return value exposes ``.name``."""
        ...


class CloudRunJobRunner:
    """Execute Cloud Run jobs and wait for their executions to finish."""

    def __init__(
        self,
        project: str,
        region: str,
        job_names: dict[PipelineJob, str],
        jobs_client: Any | None = None,
        executions_client: Any | None = None,
    ) -> None:
        """Build the Cloud Run clients once, validating the job names."""
        missing = [key for key, name in job_names.items() if not name]
        if missing:
            msg = f"Missing Cloud Run job name(s): {', '.join(missing)}"
            raise ValueError(msg)
        if jobs_client is None or executions_client is None:
            from google.cloud import run_v2  # noqa: PLC0415

            jobs_client = jobs_client or run_v2.JobsClient()
            executions_client = executions_client or run_v2.ExecutionsClient()
        self._project = project
        self._region = region
        self._job_names = job_names
        self._jobs = jobs_client
        self._executions = executions_client

    def _job_path(self, job: PipelineJob) -> str:
        return (
            f"projects/{self._project}/locations/{self._region}"
            f"/jobs/{self._job_names[job]}"
        )

    def _is_running(self, job_path: str) -> bool:
        return any(
            execution.running_count > 0
            for execution in self._executions.list_executions(parent=job_path)
        )

    def submit_and_wait(self, job: PipelineJob) -> Any:
        """Run the Cloud Run job and block until its execution finishes.

        The busy check and ``run_job`` are not atomic, so two requests can both pass
        it. The guard is best-effort across API instances; ``--max-instances 3`` and the
        job cost bound the damage. Two concurrent trainings could race on ``latest.txt``.

        Raises
        ------
        PipelineBusyError
            If an execution of the same job is still running.
        PipelineError
            If the execution failed or timed out.
        """
        job_path = self._job_path(job)
        if self._is_running(job_path):
            msg = f"{job} job is already running"
            raise PipelineBusyError(msg)
        operation = self._jobs.run_job(name=job_path)
        try:
            execution = operation.result(
                timeout=JOB_TIMEOUT_SECONDS + WAIT_MARGIN_SECONDS
            )
        except concurrent.futures.TimeoutError as exc:
            msg = f"{job} job did not finish within the wait timeout"
            raise PipelineError(msg) from exc
        if execution.failed_count or execution.succeeded_count < 1:
            msg = f"{job} job execution {execution.name} did not succeed"
            raise PipelineError(msg)
        logger.info("Cloud Run %s execution %s succeeded", job, execution.name)
        return execution


class LocalPipelineRunner:
    """Run the training/serving entry points in process (``DEPLOYMENT_MODE=local``)."""

    def __init__(self) -> None:
        """Create the lock that serialises local pipeline runs."""
        self._lock = threading.Lock()

    def submit_and_wait(self, job: PipelineJob) -> PipelineExecution:
        """Run the job in process; raise ``PipelineBusyError`` if another run holds the lock."""
        if not self._lock.acquire(blocking=False):
            msg = "a pipeline job is already running"
            raise PipelineBusyError(msg)
        try:
            from challenge import serving, training  # noqa: PLC0415

            entrypoint = training.main if job == "training" else serving.main
            try:
                entrypoint()
            except Exception as exc:
                raise PipelineError(str(exc)) from exc
        finally:
            self._lock.release()
        return PipelineExecution(name=f"local/{job}/{uuid.uuid4().hex[:8]}")


@cache
def _build_runner(mode: str) -> PipelineRunner:
    if mode == "gcp":
        settings = get_settings()
        return CloudRunJobRunner(
            project=settings.gcp_project_id,
            region=settings.gcp_region,
            job_names={
                "training": settings.cloud_run_training_job,
                "serving": settings.cloud_run_serving_job,
            },
        )
    return LocalPipelineRunner()


def get_runner() -> PipelineRunner:
    """Return the runner for the configured deployment mode (built once per mode)."""
    return _build_runner(get_settings().deployment_mode)
