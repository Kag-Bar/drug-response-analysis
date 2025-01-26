import os
import unittest
from ModelTrainer import ModelTrainer
from FeatureExtractor import FeatureExtractor

class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        """Setup any necessary data for testing."""
        self.config_path = os.path.abspath('mocks/mock_feature_extractor_cfg.json')
        self.feature_extractor = FeatureExtractor(cfg_path=self.config_path)
        self.model_trainer = ModelTrainer(self.feature_extractor)

    def test_initialize_result_dict(self):
        """Test that initialize_result_dict creates the expected dictionary structure."""
        self.model_trainer.initialize_result_dict()
        expected_keys = ["LogisticRegression_L1", "RandomForest", "XGBoost", "NeuralNetwork"]
        for model_name in expected_keys:
            self.assertIn(model_name, self.model_trainer.results)
            metrics = self.model_trainer.results[model_name]
            self.assertIsInstance(metrics, dict)
            self.assertListEqual(
                list(metrics.keys()),
                ["accuracy", "sensitivity", "specificity", "precision", "recall", "auc"]
            )
            for metric in metrics.values():
                self.assertIsInstance(metric, list)  # Ensure metrics are lists

    def test_train_and_save_model(self):
        """Test that train_and_save_model outputs a valid model and produces consistent results."""
        # Mock data for training
        x_train = [[0.1, 0.2], [1.2, 2.3], [0.3, 0.24], [4.4, 2.5]]
        y_train = [0, 1, 0, 1]

        # Train a model and check its validity
        model_name = "LogisticRegression_L1"
        model = self.model_trainer.train_and_save_model(x_train, y_train, model_name, pca=False, save_model=False)
        self.assertIsNotNone(model)  # Ensure the model is returned
        self.assertTrue(hasattr(model, "predict"))  # Check the model has a predict method
        # Test consistent predictions from the trained model
        predictions = model.predict(x_train)
        self.assertEqual(len(predictions), len(y_train))  # Ensure predictions match the input size
        expected_predictions_logistic = [1, 1, 1, 1]  # Example expected results
        self.assertListEqual(predictions.tolist(), expected_predictions_logistic)

        model_name = "RandomForest"
        model = self.model_trainer.train_and_save_model(x_train, y_train, model_name, pca=False, save_model=False)
        self.assertIsNotNone(model)  # Ensure the model is returned
        self.assertTrue(hasattr(model, "predict"))  # Check the model has a predict method
        # Test consistent predictions from the trained model
        predictions = model.predict(x_train)
        self.assertEqual(len(predictions), len(y_train))  # Ensure predictions match the input size
        expected_predictions_rf = [0, 1, 0, 1]
        self.assertListEqual(predictions.tolist(), expected_predictions_rf)

if __name__ == "__main__":
    unittest.main()
