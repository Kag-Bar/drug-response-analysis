from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
from SimpleClassificationNetwork import SimpleClassificationNetwork
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib
import json


class ModelTrainer:
    def __init__(self, FeatureExtractor):
        logging.basicConfig(level=logging.INFO)
        self.pca = None
        self.pca_n_features = None
        self.feature_extractor = FeatureExtractor
        self.nn_cfg = self.feature_extractor.cfg.get("NeuralNetwork_cfg", {})
        self.save_plots = self.feature_extractor.cfg.get("save_plots", False)
        self.output_path = self.feature_extractor.cfg.get("output_path")

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

    def train_nn_model(self, x_train, y_train):
        """Train and evaluate the neural network."""
        input_size = x_train.shape[1]  # Number of features

        # Get Training Params
        lr = self.nn_cfg.get("lr", 0.001)
        weight_decay = self.nn_cfg.get("weight_decay", 1e-4)
        num_epochs = self.nn_cfg.get("num_epochs", 10)
        num_classes = self.nn_cfg.get("num_classes", 2)

        # Convert data to PyTorch tensors
        x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

        # Initialize model, loss function, and optimizer
        model = SimpleClassificationNetwork(input_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        return model

    def eval_nn(self, model, x_test):
        x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)

        # Evaluate model
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(x_test_tensor)
            y_pred = torch.argmax(y_pred_prob, dim=1).numpy()
            return y_pred, y_pred_prob.numpy().max(axis=1)

    def train(self, cv_splits, y_col, pca=False):

        self.initialize_result_dict()
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
            nn_model = self.train_nn_model(x_train, y_train)
            self.evaluate_model(nn_model, x_test, y_test, "NeuralNetwork")

        # Average results across CV splits
        self.plot_results(len(y_train), pca, y_col)
        return self.results

    def evaluate_model(self, model, x_test, y_test, model_name):
        """Evaluate a model and store metrics."""
        if model_name == "NeuralNetwork":
            y_pred, y_pred_prob = self.eval_nn(model, x_test)
        else:
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
        self.results[model_name]["auc"].append(
            roc_auc_score(y_test, y_pred_prob) if y_pred_prob is not None else None)

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
        plt.show(block=False)
        plt.savefig(os.path.join(self.output_path, f"{title}.png"))
        plt.pause(5)
        plt.close()


    def print_results(self, results):
        """Print averaged performance metrics."""
        for model_name, metrics_list in results.items():
            avg_metrics = {k: np.mean([m[k] for m in metrics_list if m[k] is not None]) for k in metrics_list[0]}
            print(f"Model: {model_name}")
            for metric, value in avg_metrics.items():
                print(f"  {metric}: {value:.4f}")

    def plot_agumentations_impact(self, y_cols, results_dict, pca=False):
        x_labels = y_cols
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

        for idx, model_name in enumerate(results_dict[y_cols[0]]):
            acc = [np.mean(results_dict[label][model_name]['accuracy']) for label in x_labels]
            sens = [np.mean(results_dict[label][model_name]['sensitivity']) for label in x_labels]
            spec = [np.mean(results_dict[label][model_name]['specificity']) for label in x_labels]

            axs[idx].plot(x_labels, acc, label="Accuracy")
            axs[idx].plot(x_labels, sens, label="Sensitivity")
            axs[idx].plot(x_labels, spec, label="Specificity")
            axs[idx].set_ylim([0.2, 1])
            axs[idx].set_xticks(range(len(x_labels)))
            axs[idx].set_xticklabels(x_labels)
            axs[idx].set_title(f"{model_name}")
            axs[idx].legend()

        pca_suffix = "(PCA)" if pca else ""
        with open(os.path.join(self.output_path, f"results_dict{pca_suffix}.json"), "w") as outfile:
            json.dump(results_dict, outfile)

        title = f"CV Average Performances Across Different Augmentations {pca_suffix}"
        plt.suptitle(title, fontsize=16)
        plt.show(block=False)
        plt.savefig(os.path.join(self.output_path, f"{title}.png"))
        plt.pause(5)
        plt.close()

    def train_and_save_model(self, x_train, y_train, model_name, pca=False, save_model=False):
        if pca:
            x_train = self.convert_pca(x_train)

        if model_name == "LogisticRegression_L1":
            model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            model.fit(x_train, y_train)
        elif model_name == "RandomForest":
            model = RandomForestClassifier(random_state=42)
            model.fit(x_train, y_train)
        elif model_name == "XGBoost":
            model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            model.fit(x_train, y_train)
        elif model_name == "NeuralNetwork":
            model = self.train_nn_model(x_train, y_train)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Save the model
        if save_model:
            os.makedirs(self.output_path, exist_ok=True)
            model_file = os.path.join(self.output_path, f"{model_name}.model")

            if model_name in ["LogisticRegression_L1", "RandomForest", "XGBoost"]:
                joblib.dump(model, model_file)
            elif model_name == "NeuralNetwork":
                torch.save(model.state_dict(), model_file)
            else:
                raise ValueError(f"Unsupported model name for saving: {model_name}")

            print(f"Model saved to: {model_file}")
        else:
            print(f"Model had not been saved since output path is not listed in the config file")

        return model