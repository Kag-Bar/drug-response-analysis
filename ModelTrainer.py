from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleClassificationNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassificationNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),  # Input layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Hidden layer 1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),  # Hidden layer 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),  # Output layer
        )
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for probabilities

    def forward(self, x):
        x = self.model(x)
        return self.softmax(x)

def train_nn_model(x_train, x_test, y_train, y_test, input_size, num_classes):
    """Train and evaluate the neural network."""
    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Initialize model, loss function, and optimizer
    model = SimpleClassificationNetwork(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(x_test_tensor)
        y_pred = torch.argmax(y_pred_prob, dim=1).numpy()
        return y_pred, y_pred_prob.numpy()


class ModelTrainer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.pca = None
        self.pca_n_features = None

    def initialize_result_dict(self):
        self.results = {
            model_name: {"accuracy": [], "sensitivity": [], "specificity": [], "precision": [], "recall": [], "auc": []}
            for model_name in ["LogisticRegression_L1", "RandomForest", "XGBoost", "NeuralNetwork"]
        }

    def define_pca(self, pca, n_features):
        self.pca = pca
        self.pca_n_features = n_features

    def convert_pca(self, x_set):
        if not self.pca:
            logging.warning("PCA must be first defined using define_pca")

        x_pca = pd.DataFrame(
            self.pca.transform(x_set),
            columns=[f"PC{i + 1}" for i in range(self.pca_n_features)]
        )

        return x_pca

    def train(self, cv_splits, y_col, pca=False):

        self.initialize_result_dict()
        input_size = cv_splits[0][0].shape[1]  # Number of features
        num_classes = 2  # Binary case
        for x_train, x_test, y_train, y_test in cv_splits:
            if pca:
                x_train = self.convert_pca(x_train)
                x_test = self.convert_pca(x_test)

            # Logistic Regression with L1
            logistic_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            logistic_model.fit(x_train, y_train)
            self.evaluate_model(logistic_model, x_test, y_test, "LogisticRegression_L1")

            # Random Forest
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(x_train, y_train)
            self.evaluate_model(rf_model, x_test, y_test, "RandomForest")

            # XGBoost
            xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(x_train, y_train)
            self.evaluate_model(xgb_model, x_test, y_test, "XGBoost")

            # Neural Network
            y_pred, y_pred_prob = train_nn_model(x_train, x_test, y_train, y_test, input_size, num_classes)
            self.evaluate_model(None, x_test, y_test, "NeuralNetwork", y_pred, y_pred_prob)

        # Average results across CV splits
        self.plot_results(len(y_train), pca, y_col)
        return self.results

    def evaluate_model(self, model, x_test, y_test, model_name, y_pred=None, y_pred_prob=None):
        """Evaluate a model and store metrics."""
        if model:
            y_pred = model.predict(x_test)
            y_pred_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else None

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Also known as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.results[model_name]["accuracy"].append(accuracy)
        self.results[model_name]["sensitivity"].append(sensitivity)
        self.results[model_name]["specificity"].append(specificity)
        self.results[model_name]["precision"].append(precision_score(y_test, y_pred, average='binary', zero_division=0))
        self.results[model_name]["recall"].append(recall_score(y_test, y_pred, average='binary', zero_division=0))
        if model:
            self.results[model_name]["auc"].append(
                roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None)
        else:
            self.results[model_name]["auc"].append(
                roc_auc_score(y_test, y_pred_prob.max(axis=1)))

    def plot_results(self, N_y, pca=False, y_col=None):
        """Plot accuracy, sensitivity, and specificity as boxplots."""
        metrics = ["accuracy", "sensitivity", "specificity"]
        model_names = list(self.results.keys())
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:gray']

        # Prepare data for boxplots
        data = {
            metric: [self.results[model][metric] for model in model_names]
            for metric in metrics
        }

        # Set up boxplot positions
        x_ticks = np.arange(len(metrics))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, model_name in enumerate(model_names):
            positions = x_ticks + (i - 1) * width  # Adjust positions for each model
            values = [data[metric][i] for metric in metrics]
            bp = ax.boxplot(
                values,
                positions=positions,
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=colors[i], color='black'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
            )

        # Customize plot
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([metric.title() for metric in metrics])
        ax.set_ylabel("Performance Metric")
        if pca:
            title = f"Model Performance Metrics on {self.pca_n_features} PCA Features (N={N_y} with {y_col})"
        else:
            title = f"Model Performance Metrics (N={N_y} with {y_col})"
        ax.set_title(title)
        ax.legend(
            [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i]) for i in range(len(model_names))],
            model_names,
        )

        plt.tight_layout()
        plt.show()

    def print_results(self, results):
        """Print averaged performance metrics."""
        for model_name, metrics_list in results.items():
            avg_metrics = {k: np.mean([m[k] for m in metrics_list if m[k] is not None]) for k in metrics_list[0]}
            print(f"Model: {model_name}")
            for metric, value in avg_metrics.items():
                print(f"  {metric}: {value:.4f}")