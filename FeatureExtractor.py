from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.feature_selection import SelectKBest, f_classif

from DataHandler import DataHandler

class FeatureExtractor(DataHandler):
    """
    Extracts features from data and splits it into training and testing sets.
    :param cfg_path: Path to the configuration file.
    """

    def __init__(self, cfg_path):
        logging.basicConfig(level=logging.INFO)

        super().__init__(cfg_path)

        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(test_size=self.cfg.get("test_size", 0.15))
        self.n_features = self.cfg.get("features_to_extract", 10)
        self.pca_var_threh = self.cfg.get("pca_features", {}).get("pca_var_threh")

    def extract_features(self, include_xgb=False):
        """
        Extracts the most relevant features based on various methods, including ANOVA, Lasso (Logistic Regression with L1 penalty),
        Random Forest, and optionally XGBoost. The features are ranked and the top ones are selected while considering correlations.

        :param include_xgb: Whether to include XGBoost for feature selection (default: False).
        :return: List of top features.
        """
        logging.info("\nExtracting Features\n")
        k_features = min(2*self.n_features, len(self.x_train.columns.to_list()))
        # ANOVA filter
        anova_selector = SelectKBest(score_func=f_classif, k=k_features)
        anova_selector.fit(self.x_train, self.y_train)
        anova_features = self.x_train.columns[anova_selector.get_support()].tolist()

        # Lasso method (logistic regression + L1)
        logistic_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        logistic_model.fit(self.x_train, self.y_train)
        coef_importances = pd.Series(np.abs(logistic_model.coef_[0]), index=self.x_train.columns)
        lasso_features = coef_importances.nlargest(k_features).index.tolist()

        # Random Forest
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(self.x_train, self.y_train)
        rf_importances = pd.Series(rf_model.feature_importances_, index=self.x_train.columns)
        rf_features = rf_importances.nlargest(k_features).index.tolist()

        # XGBoost
        if include_xgb:
            xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(self.x_train, self.y_train)
            xgb_importances = pd.Series(xgb_model.feature_importances_, index=self.x_train.columns)
            xgb_features = xgb_importances.nlargest(k_features).index.tolist()
        else:
            xgb_features = []

        # Combine and select top 10 features with highest agreement
        feature_sets = [anova_features, lasso_features, rf_features, xgb_features, self.top_correlations]
        all_features = pd.Series([feat for sublist in feature_sets for feat in sublist])
        feature_agreement = all_features.value_counts()

        top_features = feature_agreement[feature_agreement >= 2].index.tolist()

        # If more than 10, reduce correlation
        if len(top_features) > self.n_features:
            corr_matrix = self.x_train[top_features].corr().abs()
            selected_features = []
            for feature in top_features:
                if all(corr_matrix[feature][selected_features] < 0.9):
                    selected_features.append(feature)
                    if len(selected_features) == self.n_features:
                        break
            top_features = selected_features

        # If less than 10, fill up using logistic regression ranks
        if len(top_features) < self.n_features:
            coef_importances.sort_values(ascending=False, inplace=True)
            for feat, coef in coef_importances.items():
                if coef > 0 and feat not in top_features:
                    top_features.append(feat)
                if len(top_features) == self.n_features:
                    break

        print(f"Chosen Features: {top_features}")
        return top_features

    def extract_features_pca(self):
        """
        Selects n features using PCA.
        :param n_features (int): Number of principal components to retain.
        :return:
        - transformed_x_train (pd.DataFrame): Transformed training data with n principal components.
        - transformed_x_test (pd.DataFrame): Transformed test data with n principal components.
        - explained_variance (float): Total explained variance ratio by the selected components.
        """
        logging.info("\nExtracting PCA Features\n")
        if self.pca_var_threh:
            for n in range(self.n_features,3*self.n_features+1):
                pca = PCA(n_components=n, random_state=42)
                pca.fit(pd.concat([self.x_train,self.x_test]))
                if np.sum(pca.explained_variance_ratio_) >= self.pca_var_threh:
                    break
            n_features = n
        else:
            pca = PCA(n_components=self.n_features, random_state=42)
            pca.fit(pd.concat([self.x_train,self.x_test]))
            n_features = self.n_features

        explained_variance = np.sum(pca.explained_variance_ratio_)
        joblib.dump(pca, os.path.join(self.output_path, f"pca_{n_features}_model.pkl"))

        return pca, n_features, explained_variance

