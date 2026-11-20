"""Stored-predictions queries (BigQuery in gcp mode, predictions.csv locally) with a TTL cache."""

from __future__ import annotations

import logging
import threading
from functools import cache
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from cachetools import TTLCache
from google.cloud import bigquery

from challenge.config import get_settings

logger = logging.getLogger(__name__)

PREDICTIONS_CACHE_TTL_SECONDS = 45
PREDICTIONS_CACHE_MAXSIZE = 256
PREDICTIONS_FILE = "predictions.csv"
_COLUMNS = ["OPERA", "TIPOVUELO", "MES", "prediction"]
_ORDER_BY = "OPERA, TIPOVUELO, MES, prediction"


class PredictionsRepository(Protocol):
    """Pages through stored predictions with optional OPERA/TIPOVUELO/MES filters."""

    def query(
        self,
        page: int,
        page_size: int,
        opera: list[str] | None,
        tipovuelo: list[str] | None,
        mes: list[int] | None,
    ) -> dict[str, Any]:
        """Return ``{"total_predictions": int, "predictions": [row, ...]}``."""
        ...


def _to_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "OPERA": str(record["OPERA"]),
            "TIPOVUELO": str(record["TIPOVUELO"]),
            "MES": int(record["MES"]),
            "predicted_delay": int(record["prediction"]),
        }
        for record in records
    ]


class BigQueryPredictionsRepository:
    """Query the predictions table with BigQuery query parameters only (no user SQL)."""

    def __init__(self, client: bigquery.Client) -> None:
        """Store the shared BigQuery client."""
        self._client = client

    def _table(self) -> str:
        settings = get_settings()
        return (
            f"{settings.gcp_project_id}.{settings.bq_dataset}."
            f"{settings.bq_predictions_table}"
        )

    @staticmethod
    def _filters(
        opera: list[str] | None,
        tipovuelo: list[str] | None,
        mes: list[int] | None,
    ) -> tuple[str, list[bigquery.ArrayQueryParameter]]:
        clauses: list[str] = []
        params: list[bigquery.ArrayQueryParameter] = []
        if opera is not None:
            clauses.append("OPERA IN UNNEST(@opera)")
            params.append(bigquery.ArrayQueryParameter("opera", "STRING", opera))
        if tipovuelo is not None:
            clauses.append("TIPOVUELO IN UNNEST(@tipovuelo)")
            params.append(
                bigquery.ArrayQueryParameter("tipovuelo", "STRING", tipovuelo)
            )
        if mes is not None:
            clauses.append("MES IN UNNEST(@mes)")
            params.append(bigquery.ArrayQueryParameter("mes", "INT64", mes))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def _run(self, sql: str, params: list[Any]) -> list[Any]:
        job = self._client.query(
            sql, job_config=bigquery.QueryJobConfig(query_parameters=params)
        )
        return list(job.result())

    def query(
        self,
        page: int,
        page_size: int,
        opera: list[str] | None,
        tipovuelo: list[str] | None,
        mes: list[int] | None,
    ) -> dict[str, Any]:
        """Return one page of predictions plus the total count for the same filters."""
        where, filter_params = self._filters(opera, tipovuelo, mes)
        table = self._table()
        count_rows = self._run(
            f"SELECT COUNT(*) AS n FROM `{table}`{where}",
            filter_params,
        )
        total = int(count_rows[0]["n"]) if count_rows else 0
        page_params = [
            *filter_params,
            bigquery.ScalarQueryParameter("limit", "INT64", page_size),
            bigquery.ScalarQueryParameter("offset", "INT64", (page - 1) * page_size),
        ]
        rows = self._run(
            f"SELECT OPERA, TIPOVUELO, MES, prediction FROM `{table}`{where} "
            f"ORDER BY {_ORDER_BY} LIMIT @limit OFFSET @offset",
            page_params,
        )
        records = [dict(row.items()) for row in rows]
        return {"total_predictions": total, "predictions": _to_rows(records)}


class LocalPredictionsRepository:
    """Serve predictions from ``<ARTIFACTS_DIR>/predictions.csv``, re-read when it changes."""

    def __init__(self) -> None:
        """Create the lock guarding the cached frame."""
        self._lock = threading.Lock()
        self._key: tuple[str, int] | None = None
        self._frame: pd.DataFrame = pd.DataFrame(columns=_COLUMNS)

    def _load(self) -> pd.DataFrame:
        path = Path(get_settings().artifacts_dir) / PREDICTIONS_FILE
        if not path.is_file():
            return pd.DataFrame(columns=_COLUMNS)
        key = (str(path), path.stat().st_mtime_ns)
        with self._lock:
            if key != self._key:
                self._frame = pd.read_csv(path)
                self._key = key
            return self._frame

    def query(
        self,
        page: int,
        page_size: int,
        opera: list[str] | None,
        tipovuelo: list[str] | None,
        mes: list[int] | None,
    ) -> dict[str, Any]:
        """Filter, order and page the local predictions the same way as BigQuery."""
        frame = self._load()
        mask = pd.Series(True, index=frame.index)
        if opera is not None:
            mask &= frame["OPERA"].isin(opera)
        if tipovuelo is not None:
            mask &= frame["TIPOVUELO"].isin(tipovuelo)
        if mes is not None:
            mask &= frame["MES"].isin(mes)
        matched = frame.loc[mask].sort_values(
            ["OPERA", "TIPOVUELO", "MES", "prediction"], kind="stable"
        )
        offset = (page - 1) * page_size
        page_rows = matched.iloc[offset : offset + page_size]
        return {
            "total_predictions": len(matched),
            "predictions": _to_rows(page_rows.to_dict("records")),
        }


class CachedPredictions:
    """Thread-safe TTL cache in front of a repository, keyed by the normalised arguments.

    Staleness is bounded by the TTL when the serving job runs outside the API; the API
    clears the cache itself after a successful ``POST /pipeline/predict``.
    """

    def __init__(
        self,
        inner: PredictionsRepository,
        ttl: float = PREDICTIONS_CACHE_TTL_SECONDS,
        maxsize: int = PREDICTIONS_CACHE_MAXSIZE,
    ) -> None:
        """Wrap ``inner`` with an empty cache."""
        self._inner = inner
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()
        self._generation = 0

    def query(
        self,
        page: int,
        page_size: int,
        opera: list[str] | None,
        tipovuelo: list[str] | None,
        mes: list[int] | None,
    ) -> dict[str, Any]:
        """Return a cached result when present, otherwise query and cache it."""
        key = (
            page,
            page_size,
            tuple(opera) if opera is not None else None,
            tuple(tipovuelo) if tipovuelo is not None else None,
            tuple(mes) if mes is not None else None,
        )
        with self._lock:
            cached = self._cache.get(key)
            generation = self._generation
        if cached is not None:
            return cached
        result = self._inner.query(page, page_size, opera, tipovuelo, mes)
        with self._lock:
            # Do not cache a result that was computed before an invalidation.
            if generation == self._generation:
                self._cache[key] = result
        return result

    def invalidate(self) -> None:
        """Drop every cached result."""
        with self._lock:
            self._cache.clear()
            self._generation += 1
        logger.info("Predictions cache invalidated")


@cache
def _build_repository(mode: str) -> CachedPredictions:
    if mode == "gcp":
        inner: PredictionsRepository = BigQueryPredictionsRepository(
            bigquery.Client(project=get_settings().gcp_project_id)
        )
    else:
        inner = LocalPredictionsRepository()
    return CachedPredictions(inner)


def get_repository() -> CachedPredictions:
    """Return the cached predictions repository for the configured mode (built once per mode)."""
    return _build_repository(get_settings().deployment_mode)
