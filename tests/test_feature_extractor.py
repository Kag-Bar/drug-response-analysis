import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from FeatureExtractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        # Mock configuration
        self.config_path = os.path.abspath('mocks/mock_feature_extractor_cfg.json')

        # Create mock data
        self.x_train = pd.DataFrame(np.random.rand(100, 10), columns=[f"feature_{i}" for i in range(10)])
        self.x_test = pd.DataFrame(np.random.rand(20, 10), columns=[f"feature_{i}" for i in range(10)])
        self.y_train = pd.Series(np.random.randint(0, 2, size=100))
        self.y_test = pd.Series(np.random.randint(0, 2, size=20))

        # Create a FeatureExtractor instance and mock its methods
        self.feature_extractor = FeatureExtractor(cfg_path=self.config_path)
        self.feature_extractor.x_train = self.x_train
        self.feature_extractor.x_test = self.x_test
        self.feature_extractor.y_train = self.y_train
        self.feature_extractor.y_test = self.y_test
        self.feature_extractor.output_path = "."

    def test_extract_features_pca(self):
        # Mock joblib to avoid actual file I/O
        with patch("joblib.dump") as mock_dump:
            pca, n_features, explained_variance = self.feature_extractor.extract_features_pca()

            self.assertIsInstance(pca, PCA)
            self.assertGreaterEqual(n_features, self.feature_extractor.cfg["features_to_extract"])
            self.assertGreaterEqual(explained_variance, self.feature_extractor.cfg["pca_features"]["pca_var_threh"])
            mock_dump.assert_called_once()  # Ensure PCA model is being saved

    def test_extract_features(self):
        # Test the extract_features method without XGBoost
        self.feature_extractor.top_correlations = []  # Mock any top correlation dependencies
        top_features = self.feature_extractor.extract_features(include_xgb=False)

        self.assertEqual(len(top_features), self.feature_extractor.cfg["features_to_extract"])
        self.assertTrue(all(feature in self.x_train.columns for feature in top_features))


if __name__ == "__main__":
    unittest.main()

