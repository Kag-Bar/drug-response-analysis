import os
import json
import math
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



class DataHandler:
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

        self.plot_eda = self.cfg.get("plot_EDA", True)
        self.save_plots = self.cfg.get("save_plots", False)
        self.output_path = self.cfg.get("output_path")
        self.save_data = self.cfg.get("save_data")
        self.genedata = None
        self.metadata = None
        self.data = None

        logging.info("\nLoading the CSV Dataframes\n")
        self.load_data()
        self.gene_cols = self.genedata.columns.drop("SampleID").tolist()
        self.predict_cols = ['y']

        logging.info("\nPreforming Initial Data Exploration\n")
        self.remove_all_nans()
        self.top_correlations = self.analyze_data()

        if self.cfg.get("normalized_data_path"):
            logging.info("\nLoading Pre-Normalized Data CSV Dataframe\n")
            self.data = pd.read_csv(self.cfg["normalized_data_path"], index_col=0)
            self.predict_cols = self.y_cols()
        else:
            logging.info("\nCompleteing Missing Values\n")
            self.handle_missing_values()

            logging.info("\nNormalizing The Data\n")
            self.encode_categorical()
            self.normalize_data()
            logging.info("\nFinished Normalizing Data\n")

    def load_data(self):
        """
        Load the data csv's and creates a merged data df [pandas]
        """
        # Load gene expression data
        self.genedata = pd.read_csv(self.cfg['genedata_path'], index_col=0).T
        self.genedata.index.name = "SampleID"
        self.genedata.reset_index(inplace=True)

        # Load metadata and preprocess it
        self.metadata = pd.read_csv(self.cfg['metadata_path'])
        self.preprocess_metadata()

        # Merge datasets
        self.data = self.genedata.merge(self.metadata, on='SampleID', how='inner')

    def preprocess_metadata(self):
        """
        Preprocess the meta-data by removing trivial columns, renaming columns, and binarize the response values
        :return: None (modifies self.metadata)
        """
        logging.info("Checking for single-value columns in the dataset")
        # Identify singular columns (columns with only one unique value)
        singular_cols = [col for col in self.metadata.columns if self.metadata[col].nunique() == 1]
        if singular_cols:
            if self.plot_eda:
                fig, axs = plt.subplots(1, len(singular_cols))
                fig.suptitle('Columns with a single value to be removed')
                for col_i, col in enumerate(singular_cols):
                    axs[col_i].hist(self.metadata[col])
                    axs[col_i].set_title(col.title())
                plt.show(block=False)
                plt.savefig(os.path.join(self.output_path,"Columns with a single value to be removed.png"))
                plt.pause(5)
                plt.close()

            logging.info(f"Removing columns with single value: {singular_cols}")
            self.metadata.drop(columns=singular_cols, inplace=True)
        else:
            logging.info("No singular columns found in metadata.")

        self.metadata.rename(columns={'disease activity score (das28)': 'das'}, inplace=True)
        self.metadata['y'] = self.metadata['Response status'].str.lower().map({'responder': 1, 'non_responder': 0})
        self.metadata.drop(columns=['Response status'], inplace=True)

    def normalize_data(self):
        """
        Normalize gene expression and 'das' columns based on standard z normalization.
        :return: None (modifies self.data)
        """
        if self.cfg.get('norm_genes', False):
            scaler = StandardScaler()
            self.data[self.gene_cols] = scaler.fit_transform(self.data[self.gene_cols])

        if 'das' in self.data.columns:
            self.data['das'] = (self.data['das'] - self.data['das'].mean(skipna=True)) / self.data['das'].std(skipna=True)

        if self.save_data:
            self.data.to_csv(os.path.join(self.output_path, 'data_normalized.csv'))

    def remove_all_nans(self, rows_percent_to_remove=0.15):
        """
        :param rows_percent_to_remove (float): A percentage of rows that are willing to be removed if both
            response value and disease score are missing
        :return: None (modifies self.data)
        """
        all_nan_rows = len(self.data[self.data['y'].isna() & self.data['das'].isna()])
        if all_nan_rows < rows_percent_to_remove*len(self.data):
            logging.info(f"Removing {all_nan_rows} rows with missing Response and disease score")
            self.data = self.data[~(self.data['y'].isna() & self.data['das'].isna())].reset_index(drop=True)

    def analyze_data(self):
        """
        Analyzes and visualizes initial data distribution and correlations.
        :return: top_correlations (list) ranked 20 highest absoult correlation
        """
        if self.plot_eda:
            # Visualize response distribution
            self.data.assign(
                y=self.data['y'].fillna('NaN').replace({0: False, 1: True})
            ).groupby(
                ['y', 'Gender']
            ).size().unstack(fill_value=0).plot(
                kind='bar', edgecolor='k'
            )
            plt.title('Response Status Distribution')
            plt.ylabel('Count')
            plt.xlabel('Response Status')
            plt.show(block=False)
            plt.savefig(os.path.join(self.output_path, "Response Status Distribution.png"))
            plt.pause(5)
            plt.close()

            # Visualize DAS vs Response status
            self.data.fillna('NaN').replace({0: False, 1: True}).boxplot(column='das', by='y', grid=True)
            plt.title('Disease Activity Score vs Response Status')
            plt.xlabel('Response Status')
            plt.ylabel('DAS')
            plt.suptitle('')  # Removes extra suptitle generated by boxplot
            plt.show(block=False)
            plt.savefig(os.path.join(self.output_path, "Disease Activity Score vs Response Status.png"))
            plt.pause(5)
            plt.close()

        # Calculate gene correlation
        logging.info(f"Exploring genes correlation (calculating correlation, may take some time)")
        correlation = self.data[self.gene_cols].corrwith(self.data['y'])
        top_correlations = pd.concat([
            correlation[correlation > 0].sort_values(ascending=False).head(10),
            correlation[correlation < 0].sort_values().head(10)
        ])

        if self.plot_eda:
            # Visualize top correlations with scatter plots
            fig, axes = plt.subplots(5, 4)
            axes = axes.flatten()
            plt.subplots_adjust(hspace=0.8, wspace=0.4)

            for i, (col, corr) in enumerate(top_correlations.items()):
                ax = axes[i]
                sorted_values = self.data[col].sort_values()
                sorted_y = self.data.loc[sorted_values.index, 'y']

                scatter = ax.scatter(range(len(sorted_values)), sorted_values, c=sorted_y, cmap='coolwarm', edgecolors='k')
                ax.set_title(f'{col} (Correlation: {corr:.2f})')
                ax.set_xticks([])  # Remove x-axis ticks for clarity
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('y Value')
            plt.suptitle('Gene Correlation')
            plt.show(block=False)
            plt.savefig(os.path.join(self.output_path, "Initial Gene Correlation.png"))
            plt.pause(10)
            plt.close()

        return top_correlations.index.to_list()

    def handle_missing_values(self, k=3, low_mse=0.15, high_mse=0.25):
        # 1. Separate rows where y is NaN and where y is not NaN
        data_non_nan = self.data[~self.data['y'].isna()]  # Rows where y is not NaN
        data_nan = self.data[self.data['y'].isna()]  # Rows where y is NaN

        # 2. Extract features (Genes cols only)
        features_non_nan = data_non_nan[self.gene_cols]
        features_non_nan = features_non_nan.to_numpy().astype(np.float64)
        features_nan = data_nan[self.gene_cols]

        # 3. Initialize a list to store results
        nearest_neighbors = []

        # Initialize the Nearest Neighbors model
        knn = NearestNeighbors(n_neighbors=k, metric='euclidean')  # Euclidean distance for KNN
        knn.fit(features_non_nan)

        # 4. Calculate Cosine Similarity and MSE for each row where y is NaN
        for index, row_nan in features_nan.iterrows():
            row_nan_vector = row_nan.to_numpy().astype(np.float64).reshape(1, -1)
            cosine_similarities = cosine_similarity(row_nan_vector, features_non_nan)

            # Calculate MSE between row_nan and all rows in data_non_nan
            mse_values = [mean_squared_error(row_nan_vector, row.reshape(1, -1)) for row in features_non_nan]

            # Find the index of the nearest row based on Cosine Similarity and MSE
            nearest_cosine_idx = np.argmax(cosine_similarities)
            nearest_mse_idx = np.argmin(mse_values)

            # Get the corresponding values
            nearest_cosine_row = data_non_nan.iloc[nearest_cosine_idx]
            nearest_mse_row = data_non_nan.iloc[nearest_mse_idx]

            nearest_cosine_value = cosine_similarities[0, nearest_cosine_idx]
            nearest_mse_value = mse_values[nearest_mse_idx]

            # KNN
            distances, indices = knn.kneighbors([row_nan_vector.flatten()], n_neighbors=1)
            knn_nearest = data_non_nan.iloc[indices[0]]['SampleID'].values[0]

            # Add results to the list
            nearest_neighbors.append({
                'nan_row': data_nan.loc[index, 'SampleID'],
                'nearest_cos': nearest_cosine_row['SampleID'],
                'cos_sim': nearest_cosine_value,
                'nearest_mse': nearest_mse_row['SampleID'],
                'mse': nearest_mse_value,
                'nearest_knn': knn_nearest
            })

        # Convert the results into a DataFrame
        nearest_neighbors_df = pd.DataFrame(nearest_neighbors)

        nearest_neighbors_df.sort_values(by=['cos_sim','mse'], ascending=[False, True], inplace=True)
        if self.plot_eda:
            plt.figure()
            for i, row in nearest_neighbors_df.iterrows():
                # Check if nearest row by MSE and Cosine Similarity are the same
                if row['nearest_mse'] == row['nearest_cos'] and row['nearest_mse'] == row['nearest_knn']:
                    # Plot as a big asterisk
                    plt.scatter(row['cos_sim'], row['mse'], color='tab:blue', s=100, marker='*',
                                label='Same Nearest Row')
                    if row['mse'] < low_mse:
                        plt.scatter(row['cos_sim'], row['mse'], color='tab:green', s=500, marker='o', alpha=0.2)
                    elif row['mse'] < high_mse:
                        plt.scatter(row['cos_sim'], row['mse'], color='tab:olive', s=500, marker='o', alpha=0.2)
                else:
                    # Regular scatter plot
                    plt.scatter(row['cos_sim'], row['mse'], color='tab:red', alpha=0.6)

            # Add labels and title
            plt.title('Cosine Similarity vs MSE')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('MSE')
            plt.show(block=False)
            plt.savefig(os.path.join(self.output_path, "Cosine Similarity vs MSE.png"))
            plt.pause(5)
            plt.close()

        nearest_neighbors_df = nearest_neighbors_df[
            (nearest_neighbors_df['nearest_mse'] == nearest_neighbors_df['nearest_cos']) &
            (nearest_neighbors_df['nearest_cos'] == nearest_neighbors_df['nearest_knn']) &
            (nearest_neighbors_df['mse'] <= high_mse)
            ]

        self.data[f'y_augment_{low_mse}'] = np.nan
        self.data[f'y_augment_{high_mse}'] = np.nan
        for i, row in self.data.iterrows():
            if math.isnan(row['y']):
                if row['SampleID'] in nearest_neighbors_df.nan_row.tolist():
                    nearest_neighbors_row = nearest_neighbors_df[nearest_neighbors_df.nan_row == row['SampleID']]
                    nearest = nearest_neighbors_row['nearest_mse'].values[0]
                    if nearest_neighbors_row['mse'].values[0] <= high_mse:
                        self.data.at[i, f'y_augment_{high_mse}'] = self.data[self.data['SampleID'] == nearest]['y'].values[0]
                    if nearest_neighbors_row['mse'].values[0] <= low_mse:
                        self.data.at[i, f'y_augment_{low_mse}'] = self.data[self.data['SampleID'] == nearest]['y'].values[0]
            else:
                self.data.at[i, f'y_augment_{high_mse}'] = row['y']
                self.data.at[i, f'y_augment_{low_mse}'] = row['y']

        self.predict_cols.extend([f'y_augment_{low_mse}', f'y_augment_{high_mse}'])

    def visualize_gene_distribution(self, gene):
        """
        Visualize the distribution of a specific gene's expression levels.
        :param gene: Gene symbol (string) to visualize.
        """
        if gene not in self.genedata.columns:
            raise ValueError(f"Gene {gene} not found in the dataset.")

        plt.figure(figsize=(8, 5))
        plt.hist(self.genedata[gene], bins=30, color='blue', alpha=0.7)
        plt.title(f'{gene} Expression Distribution')
        plt.xlabel('Expression Level')
        plt.ylabel('Frequency')
        plt.show()
        plt.savefig(os.path.join(self.output_path, f'{gene} Expression Distribution.png'))
        plt.close()

    def train_test_split(self, y_col='y', feature_cols=None, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        :param test_size: Proportion of the data to include in the test split.
        :param random_state: Random seed for reproducibility.
        """
        if not feature_cols:
            feature_cols = self.data.columns.to_list()
            feature_cols = [feat for feat in feature_cols if feat not in self.predict_cols and 'ID' not in feat]

        X = self.data[self.data[y_col].notna()][feature_cols]
        y = self.data[self.data[y_col].notna()][y_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def cross_val_split(self, y_col='y', feature_cols=None, n_splits=5, random_state=42):
        """
        Split the data into cross-validation sets.
        :param n_splits: Number of splits for cross-validation.
        :param random_state: Random seed for reproducibility.
        """
        if not feature_cols:
            feature_cols = self.data.columns.to_list()
            feature_cols = [feat for feat in feature_cols if feat not in self.predict_cols and 'ID' not in feat]

        X = self.data[self.data[y_col].notna()][feature_cols]
        y = self.data[self.data[y_col].notna()][y_col]

        # Initialize KFold with the specified number of splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Create a list to store train-test splits
        cv_splits = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            cv_splits.append((X_train, X_test, y_train, y_test))

        return cv_splits

    def y_cols(self):
        return self.data.columns[self.data.columns.get_loc("y"):].to_list()

    def encode_categorical(self):
        if 'Gender' in self.data.columns:
            encoder = LabelEncoder()
            self.data['Gender'] = encoder.fit_transform(self.data['Gender'])
