import unittest

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from challenge.model import DelayModel


class TestModel(unittest.TestCase):  # noqa: D101
    FEATURES_COLS = [  # noqa: RUF012
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

    TARGET_COL = [  # noqa: RUF012
        "delay"
    ]

    def setUp(self) -> None:
        """Initialize model and load test data."""
        super().setUp()
        self.model = DelayModel()
        self.data = pd.read_csv(filepath_or_buffer="../data/data.csv")

    def test_model_preprocess_for_training(self) -> None:
        """Test preprocess returns correct features and target for training."""
        features, target = self.model.preprocess(data=self.data, target_column="delay")

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

        assert isinstance(target, pd.DataFrame)
        assert target.shape[1] == len(self.TARGET_COL)
        assert set(target.columns) == set(self.TARGET_COL)

    def test_model_preprocess_for_serving(self) -> None:
        """Test preprocess returns correct features for serving without target."""
        features = self.model.preprocess(data=self.data)

        assert isinstance(features, pd.DataFrame)
        assert features.shape[1] == len(self.FEATURES_COLS)
        assert set(features.columns) == set(self.FEATURES_COLS)

    def test_model_fit(self) -> None:
        """Test model fit trains and meets performance thresholds."""
        features, target = self.model.preprocess(data=self.data, target_column="delay")

        _, features_validation, _, target_validation = train_test_split(
            features, target, test_size=0.33, random_state=42
        )

        self.model.fit(features=features, target=target)

        predicted_target = self.model._model.predict(features_validation)

        report = classification_report(
            target_validation, predicted_target, output_dict=True
        )

        recall_threshold_0 = 0.60
        f1_threshold_0 = 0.70
        recall_threshold_1 = 0.60
        f1_threshold_1 = 0.30

        assert report["0"]["recall"] < recall_threshold_0
        assert report["0"]["f1-score"] < f1_threshold_0
        assert report["1"]["recall"] > recall_threshold_1
        assert report["1"]["f1-score"] > f1_threshold_1

    def test_model_predict(self) -> None:
        """Test predict returns correct format and length."""
        features = self.model.preprocess(data=self.data)

        predicted_targets = self.model.predict(features=features)

        assert isinstance(predicted_targets, list)
        assert len(predicted_targets) == features.shape[0]
        assert all(
            isinstance(predicted_target, int) for predicted_target in predicted_targets
        )
