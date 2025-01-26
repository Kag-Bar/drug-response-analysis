import json
import logging
from sklearn.model_selection import train_test_split

from FeatureExtractor import FeatureExtractor

class InferenceFeatureExtractor(FeatureExtractor):
    """
    Initializes the InferenceFeatureExtractor class by loading a configuration file and setting up the relevant settings.
    :param cfg_path: Path to the configuration file (must be a string).
    """
    def __init__(self, cfg_path):
        logging.basicConfig(level=logging.INFO)

        if not isinstance(cfg_path, str):
            raise ValueError(f"{cfg_path} must be a string pointing to the configuration file")
        try:
            with open(cfg_path, 'r') as file:
                self.cfg = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {cfg_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from the file: {cfg_path}")

        self.save_data, self.plot_eda = False, False
        self.load_data()

    def prepare_data(self, training_or_inference='inference'):
        """
        Prepares the data for either training or inference based on the configuration.
        :param training_or_inference: Specifies whether the preparation is for 'training' or 'inference'.
        :returns:
            - For 'training': Returns the training and testing data splits (X_train, X_test, y_train, y_test).
            - For 'inference': Returns the prepared data for inference.
        """
        if self.cfg.get("use_pca", False):
            self.remove_all_nans()
            self.encode_categorical()
            self.normalize_data()
        else:
            if training_or_inference == "training":
                self.data = self.data[self.cfg['chosen_features'] + ["y"]]
            else:
                self.data = self.data[self.cfg['chosen_features']]
            self.normalize_data(columns=self.cfg['chosen_features'])

        if training_or_inference == "training":
            X_train, X_test, y_train, y_test = self.train_test_split(y_col='y', test_size=self.cfg.get("test_size", 0.15))
            return X_train, X_test, y_train, y_test
        else:
            return self.data

    def train_test_split(self, y_col='y', test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        :param test_size: Proportion of the data to include in the test split.
        :param random_state: Random seed for reproducibility.
        """

        X = self.data[self.data[y_col].notna()].drop(columns=[y_col])
        y = self.data[self.data[y_col].notna()][y_col]

        if test_size == 0:
            X_train, X_test, y_train, y_test = X, None, y, None
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        return X_train, X_test, y_train, y_test
