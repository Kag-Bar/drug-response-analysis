from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


from DataHandler import DataHandler

class FeatureExtractor(DataHandler):
    def __init__(self, cfg_path):
        super().__init__(cfg_path)

        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(test_size=0.15)
        self.encode_categorical()

    def encode_categorical(self):
        if 'Gender' in self.x_train.columns:
            encoder = LabelEncoder()
            self.x_train['Gender'] = encoder.fit_transform(self.x_train['Gender'])
            self.x_test['Gender'] = encoder.transform(self.x_test['Gender'])

    def extract_features(self, n_features=10):
        # Step 1: ANOVA filter
        anova_selector = SelectKBest(score_func=f_classif, k=2*n_features)
        anova_selector.fit(self.x_train, self.y_train)
        anova_features = self.x_train.columns[anova_selector.get_support()].tolist()

        # Step 2: Lasso method (logistic regression + L1)
        logistic_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        logistic_model.fit(self.x_train, self.y_train)
        coef_importances = pd.Series(np.abs(logistic_model.coef_[0]), index=self.x_train.columns)
        lasso_features = coef_importances.nlargest(2*n_features).index.tolist()

        # Step 4: Hybrid Method (Random Forest + XGBoost)
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(self.x_train, self.y_train)
        rf_importances = pd.Series(rf_model.feature_importances_, index=self.x_train.columns)
        rf_features = rf_importances.nlargest(2*n_features).index.tolist()

        xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(self.x_train, self.y_train)
        xgb_importances = pd.Series(xgb_model.feature_importances_, index=self.x_train.columns)
        xgb_features = xgb_importances.nlargest(2*n_features).index.tolist()

        # Combine and select top 10 features with highest agreement
        feature_sets = [anova_features, lasso_features, rf_features, xgb_features, self.top_correlations]
        all_features = pd.Series([feat for sublist in feature_sets for feat in sublist])
        feature_agreement = all_features.value_counts()

        top_features = feature_agreement[feature_agreement >= 2].index.tolist()

        # If more than 10, reduce correlation
        if len(top_features) > n_features:
            corr_matrix = self.x_train[top_features].corr().abs()
            selected_features = []
            for feature in top_features:
                if all(corr_matrix[feature][selected_features] < 0.9):
                    selected_features.append(feature)
                    if len(selected_features) == n_features:
                        break
            top_features = selected_features

        # If less than 10, fill up using logistic regression ranks
        if len(top_features) < n_features:
            coef_importances.sort_values(ascending=False, inplace=True)
            for feat, coef in coef_importances.items():
                if coef > 0 and feat not in top_features:
                    top_features.append(feat)
                if len(top_features) == n_features:
                    break

        return top_features

    def extract_features_pca(self, n_features=10, var_threh=None):
        """
        Selects n features using PCA.
        :param n_features (int): Number of principal components to retain.
        :return:
        - transformed_x_train (pd.DataFrame): Transformed training data with n principal components.
        - transformed_x_test (pd.DataFrame): Transformed test data with n principal components.
        - explained_variance (float): Total explained variance ratio by the selected components.
        """
        if var_threh:
            for n in range(n_features,3*n_features):
                pca = PCA(n_components=n, random_state=42)
                pca.fit(self.x_train)
                if np.sum(pca.explained_variance_ratio_) >= var_threh:
                    break
                    n_features = n
        else:
            pca = PCA(n_components=n_features, random_state=42)
            pca.fit(self.x_train)

        # Transform the training and test data
        transformed_x_train = pd.DataFrame(
            pca.transform(self.x_train),
            columns=[f"PC{i + 1}" for i in range(n_features)]
        )
        transformed_x_test = pd.DataFrame(
            pca.transform(self.x_test),
            columns=[f"PC{i + 1}" for i in range(n_features)]
        )

        explained_variance = np.sum(pca.explained_variance_ratio_)

        return transformed_x_train, transformed_x_test, explained_variance

# Example usage:
extractor = FeatureExtractor(cfg_path='E:\Studies\pythonProject\drug-response-analysis\cfg.json')
selected_features = extractor.extract_features()
pca_x_train, pca_x_test, variance = extractor.select_features_with_pca(n_features=10)
print(f"Total Explained Variance: {variance:.2%}")

print(selected_features)

