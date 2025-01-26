import os
import json
import torch
import joblib
from ModelTrainer import ModelTrainer
from SimpleClassificationNetwork import SimpleClassificationNetwork

class ModelTest(ModelTrainer):
    """
    Class for testing models, inheriting from ModelTrainer.
    :param FeatureExtractor: The feature extractor object used to extract features for the model.
    """
    def __init__(self, FeatureExtractor):
        super().__init__(FeatureExtractor)
        if self.feature_extractor.cfg.get("pca", {}).get("n_features"):
            self.input_size = self.feature_extractor.cfg.get("pca", {}).get("n_features")
        else:
            self.input_size = len(self.feature_extractor.cfg.get("chosen_features", []))

        self.num_classes = 2 # Binary classification

    def load_model(self, model_name, model_path):
        """
        Loads a saved model.
        :param model_name: Name of the model to load (e.g., "LogisticRegression_L1", "RandomForest").
        :param model_path: The path to the saved model.
        """
        self.model_name = model_name
        # Load and run the model
        if model_name == "NeuralNetwork":
            # Load the neural network model
            self.model = self.initialize_nn_model()  # You need a method to define the NN architecture
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

        elif model_name in ["LogisticRegression_L1", "RandomForest", "XGBoost"]:
            # Load the model using joblib
            self.model = joblib.load(model_path)

    def initialize_nn_model(self):
        """
        Initializes the SimpleClassificationNetwork with appropriate input size and number of classes.

        :return: An instance of SimpleClassificationNetwork.
        """
        return SimpleClassificationNetwork(input_size=self.input_size, num_classes=self.num_classes)

    def run_model(self, X, pca_flag=False):
        """
        Runs the model on the provided data and saves the predictions to a JSON file.

        :param X: Input feature matrix for prediction.
        :param pca_flag: Flag to apply PCA transformation (default: False).
        :returns: A tuple containing predicted labels (y_pred) and predicted probabilities (y_pred_prob).
        """
        if pca_flag:
            X = self.convert_pca(X)

        if self.model_name == "NeuralNetwork":
            y_pred, y_pred_prob = self.eval_nn(self.model, X)
        else:
            y_pred = self.model.predict(X)
            y_pred_prob = self.model.predict_proba(X)[:, 1] if hasattr(self.model, "predict_proba") else None

        # Convert numpy arrays to Python lists for JSON serialization
        predictions = {
            "y_pred": y_pred.tolist(),
            "y_pred_prob": y_pred_prob.tolist()
        }

        # Save as JSON
        with open(os.path.join(self.output_path, f"{self.model_name}_output.json"), "w") as json_file:
            json.dump(predictions, json_file, indent=4)

        return y_pred, y_pred_prob


