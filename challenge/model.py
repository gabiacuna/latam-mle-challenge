from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import xgboost as xgb

from challenge.connectors.local import LocalArtifactStore, LocalCSVClient
from challenge.connectors.protocols import (
    ArtifactStore,
    DataReader,
)

FEATURES_COLS: list[str] = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]
CATEGORICAL_COLS: list[str] = ["OPERA", "TIPOVUELO", "MES"]
SCHEDULED_COL = "Fecha-I"
OPERATED_COL = "Fecha-O"
DELAY_THRESHOLD_MINUTES = 15
XGB_RANDOM_STATE = 1
XGB_LEARNING_RATE = 0.01


def compute_min_diff(data: pd.DataFrame) -> pd.Series:
    """Return minutes between operated (Fecha-O) and scheduled (Fecha-I) departure.

    Unparseable timestamps yield NaN instead of raising.
    """
    scheduled = pd.to_datetime(data[SCHEDULED_COL], errors="coerce")
    operated = pd.to_datetime(data[OPERATED_COL], errors="coerce")
    return (operated - scheduled).dt.total_seconds() / 60


def get_period_day(timestamps: pd.Series) -> pd.Series:
    """Classify scheduled timestamps as ``mañana``, ``tarde`` or ``noche``.

    Hour-based so every instant is covered (the notebook version left gaps
    such as 11:59:30 and returned ``None``).
    """
    hours = pd.to_datetime(timestamps, errors="coerce").dt.hour
    period = pd.Series("noche", index=hours.index, dtype="object")
    period[(hours >= 5) & (hours < 12)] = "mañana"  # noqa: PLR2004
    period[(hours >= 12) & (hours < 19)] = "tarde"  # noqa: PLR2004
    period[hours.isna()] = None
    return period


def is_high_season(timestamps: pd.Series) -> pd.Series:
    """Flag scheduled dates inside high-season ranges (inclusive, whole days).

    The notebook compared full datetimes with midnight range ends, which
    excluded every flight after 00:00 on the last day of a range.
    """
    dates = pd.to_datetime(timestamps, errors="coerce")
    month_day = dates.dt.month * 100 + dates.dt.day
    ranges = [(1215, 1231), (101, 303), (715, 731), (911, 930)]
    flag = pd.Series(False, index=dates.index)
    for start, end in ranges:
        flag |= (month_day >= start) & (month_day <= end)
    return flag.astype(int)


class BaseDelayModel(ABC):
    """Abstract interface — defines the obligatory ML contract.

    Subclasses must implement preprocess, fit, and predict.
    Attempting to instantiate DelayModel without all three raises TypeError.
    """

    def __init__(self) -> None:
        self._model = None  # fitted XGBClassifier, populated by fit()
        self._run_id: str | None = None  # run id of the loaded artifact, set by load()

    @property
    def run_id(self) -> str | None:
        """Run id of the artifact currently loaded into the model, if any."""
        return self._run_id

    @abstractmethod
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or inference.

        Parameters
        ----------
        data : pd.DataFrame
            Raw flight records with columns needed for training and/or inference.
        target_column : str, optional
            Name of the target column to return (e.g. "delay").

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Features and target DataFrames when target_column is provided.
        pd.DataFrame
            Features DataFrame only when target_column is None.
        """

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train the selected model.

        Parameters
        ----------
        features : pd.DataFrame
            Preprocessed feature matrix (columns = FEATURES_COLS).
        target : pd.DataFrame
            Binary delay labels (column = "delay").
        """

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> list[int]:
        """Return binary delay predictions for each row in features.

        Parameters
        ----------
        features : pd.DataFrame
            Preprocessed feature matrix (columns = FEATURES_COLS).

        Returns
        -------
        list[int]
            0 (no delay) or 1 (delayed) for each input row.
        """


class GCPModelMixin:
    """Optional GCP I/O operations with injectable client support.

    Accepts ``reader`` and ``store`` keyword arguments so local and GCP
    implementations can be swapped without changing the step-function signatures.
    Defaults to ``LocalCSVReader`` and ``LocalArtifactStore`` for zero-credential
    local operation.
    """

    def __init__(
        self,
        reader: DataReader | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        """Initialise with optional injectable clients.

        Parameters
        ----------
        reader : DataReader, optional
            Data-loading client; defaults to ``LocalCSVReader()``.
        store : ArtifactStore, optional
            Artifact-persistence client; defaults to ``LocalArtifactStore()``.
        """
        self._reader: DataReader = reader if reader is not None else LocalCSVClient()
        self._store: ArtifactStore = (
            store if store is not None else LocalArtifactStore()
        )

    def load_data(self, project: str = "", bq_dataset: str = "") -> pd.DataFrame:
        """Load raw flight data via the injected DataReader.

        Parameters
        ----------
        project : str
            GCP project ID (unused in local mode; accepted for API compatibility).
        bq_dataset : str
            BigQuery dataset name (unused in local mode; accepted for API compatibility).

        Returns
        -------
        pd.DataFrame
            Raw flight records ready to pass to preprocess().
        """
        return self._reader.read()

    def save(self, bucket: str = "", run_id: str = "") -> str:
        """Persist the fitted model via the injected ArtifactStore.

        Parameters
        ----------
        bucket : str
            GCS bucket name (unused in local mode; accepted for API compatibility).
        run_id : str
            Unique identifier for this training run.

        Returns
        -------
        str
            Path or URI of the saved artifact.
        """
        return self._store.save(self, run_id)

    def load(self, bucket: str = "", run_id: str | None = None) -> None:
        """Restore a model via the injected ArtifactStore.

        Parameters
        ----------
        bucket : str
            GCS bucket name (unused in local mode; accepted for API compatibility).
        run_id : str, optional
            Specific run to load; latest artifact when omitted.
        """
        self._store.load(self, run_id)


class DelayModel(GCPModelMixin, BaseDelayModel):
    """Concrete class to implement for the challenge.

    Required ML methods are declared abstract in BaseDelayModel — omitting
    any of them raises TypeError at instantiation time.

    GCP I/O methods from GCPModelMixin are optional: implement them to
    connect preprocess/fit/predict to BigQuery and Cloud Storage.
    """

    def __init__(
        self,
        reader: DataReader | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        """Initialise the model and its injectable I/O clients."""
        BaseDelayModel.__init__(self)
        GCPModelMixin.__init__(self, reader=reader, store=store)

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or inference.

        Features are the one-hot encoding of OPERA, TIPOVUELO and MES, restricted
        to ``FEATURES_COLS``. Dummies absent from ``data`` are filled with 0 so
        serving batches (which carry no dates) always have the training layout.

        Raises
        ------
        ValueError
            If a required column is missing.
        """
        missing = [c for c in CATEGORICAL_COLS if c not in data.columns]
        if missing:
            msg = f"Missing required columns: {missing}"
            raise ValueError(msg)

        target: pd.DataFrame | None = None
        if target_column is not None:
            missing = [
                c for c in (SCHEDULED_COL, OPERATED_COL) if c not in data.columns
            ]
            if missing:
                msg = f"Missing columns needed to compute '{target_column}': {missing}"
                raise ValueError(msg)
            min_diff = compute_min_diff(data)
            valid = min_diff.notna()
            data = data.loc[valid]
            target = (
                (min_diff[valid] > DELAY_THRESHOLD_MINUTES)
                .astype(int)
                .to_frame(target_column)
            )

        features = pd.get_dummies(
            data[CATEGORICAL_COLS].astype({"MES": "int64"}, errors="ignore"),
            columns=CATEGORICAL_COLS,
            prefix=CATEGORICAL_COLS,
        ).reindex(columns=FEATURES_COLS, fill_value=0)
        features = features.astype(int).reset_index(drop=True)

        if target is None:
            return features
        return features, target.reset_index(drop=True)

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train an XGBoost classifier with class-imbalance weighting.

        ``scale_pos_weight`` is the negative/positive ratio of the training target.
        """
        labels = target.iloc[:, 0] if isinstance(target, pd.DataFrame) else target
        n_positive = int((labels == 1).sum())
        n_negative = int((labels == 0).sum())
        scale = n_negative / n_positive if n_positive else 1.0
        self._model = xgb.XGBClassifier(
            random_state=XGB_RANDOM_STATE,
            learning_rate=XGB_LEARNING_RATE,
            scale_pos_weight=scale,
        )
        self._model.fit(features[FEATURES_COLS], labels)

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Return binary delay predictions.

        If no model is fitted yet, the latest persisted artifact is loaded first.
        """
        if self._model is None:
            self.load()
        return [int(p) for p in self._model.predict(features[FEATURES_COLS])]
