# drug-response-analysis
Given a dataset containing gene expression data and patient metadata from a clinical trial evaluating a new drug for an autoimmune disease, the repository providing a full EDA and a classification model to predict samples treatment response.

### General Repository Structure

#### Configurations:
- **cfg**:  
  General configuration for analysis. Key parameters include:  
  - `metadata_path`: Path to the metadata table (must).  
  - `genedata_path`: Path to the genes table (must).  
  - `output_path`: Directory for saving results and analyses.  
  - `normalized_data_path`: Path to the pre-normalized data table for runtime efficiency.  
  - `norm_genes`: Boolean flag for normalizing gene expressions.  
  - `plot_EDA`: Boolean flag for performing and visualizing exploratory data analysis (EDA).  

- **cfg_train**:  
  Extends the general configuration with additional parameters:  
  - `chosen_features`: Features selected from prior analysis to reduce runtime.  
  - `use_pca`: Boolean flag for applying PCA for dimensionality reduction.  
  - `pca_path`: Path to the pre-trained PCA model.  

- **cfg_test**:  
  - `models_path`: Dictionary mapping model names to their corresponding pre-trained model paths.

#### Scripts:
- **main_research**:  
  Primary script for analysis. Performs research with/without PCA, tests various methods for handling missing data, and evaluates multiple models. Saves JSON results in the output directory, including selected features or PCA model results (if applicable).  

- **main_train**:  
  Trains models using specified model names and pre-selected features and/or a pre-trained PCA model. Saves the models and evaluates performance on the test set.  

- **main_test**:  
  Runs inference on input data using pre-trained models provided by their paths. Assumes data has no ground truth and does not perform evaluation. Saves predictions as a JSON file.  

#### Objects:
- **DataHandler**:  
  Prepares and analyzes data, including loading, missing value imputation, feature removal, normalization, and correlation analysis.  

- **FeatureExtractor**:  
  Extracts features for training using a combination of filter methods, model-based techniques, and/or dimensionality reduction methods.  

- **InferenceFeatureExtractor**:  
  Prepares data and extracts features specifically for evaluation and testing.  

- **SimpleClassificationNetwork**:  
  A basic classification network object.  

- **ModelTrainer**:  
  Trains and evaluates models on training/testing datasets with optional PCA. Limited to four predefined models.  

- **ModelTest**:  
  Loads pre-trained models and performs inference on input data without evaluation.  

#### Tests:
Contains a folder with a set of unit tests for standard input-output validations and more.


## Methods and Analysis

### 1. DataHandler - Key insights from EDA

**a. Loading the data**  
The model expects the metadata to include at least three columns with the names:  
`SampleID`, `disease activity score (das28)`, and `Response status`.

**b. Gene filtering**  
Genes from samples in `genedata` that do not appear in the metadata will not be included in the analysis (since no prediction and/or DAS value exists for them).

**c. Removing non-informative columns**  
At the beginning of the analysis, columns with no informative data are removed. These are columns where all samples have the same value (e.g., tissue type, protocol, etc.).  
![Columns with a single value to be removed](https://github.com/user-attachments/assets/735d8533-c20e-402d-bf04-51a2cbe3f3c8)

**d. Removing unreliable data points**  
Data points for which both `disease activity score (das28)` and `Response status` are missing are removed from the dataset. These data points are unreliable because no expert collected information about these samples. In total, 6 points are removed.

**e. Initial response distribution analysis**  
In an initial analysis, I examined the distribution of the drug response among the samples, separated by gender. I noticed that the data is distributed fairly evenly both in terms of response and gender. Therefore, there was no need for stratification, and normal weighting was applied to the data points during training and dataset splitting.  
![Response Status Distribution](https://github.com/user-attachments/assets/b15d056c-9fb3-4eda-af67-7ce41fa19b37)


**f. Correlation between DAS and response**  
A direct correlation between DAS and response is analyzed under the assumption that DAS has a high correlation with the response. However, it is evident that the distribution of DAS values across the responses is not significant.  
![Disease Activity Score vs Response Status](https://github.com/user-attachments/assets/31217333-d905-48b5-b4b8-4f61440b2648)


**g. Handling missing values**  
A study of the effect of filling in missing values on the model results is conducted.  
Data is not normalized here, to preserve the magnitude of gene expression values.
Data points with unknown responses are matched to data points with known responses using three methods:  

- **MSE distance** from other points.  
- **Cosine similarity** more suitable for high-dimensional vectors, but can vary in size (hence the MSE)
- **Nearest neighbor** using the KNN method.  

A graph showing MSE values versus cosine similarity values is generated. As expected, a generally linear response is observed.  
![Cosine Similarity vs MSE](https://github.com/user-attachments/assets/f3a7528f-b813-4ba8-a394-57b49c6f1936)


Data points for which all three methods identify the same sample as the closest neighbor are marked with a star. It can be seen that most missing values achieve high alignment.  

Missing values are tested in two rounds:  
1. Filling values for MSE < 0.15 (colored in green).  
2. Filling values for MSE < 0.25 (colored in yellow).  
These thresholds are chosen based on the graph described above.  

The missing responses are filled based on the response of the closest neighbor, and the impact will be discussed later.

**h. Normalizing the data**  
Later, the data is normalized (per gene across all samples). A pre-normalized table can also be loaded, as normalizing ~60K columns can be time-consuming.

**i. Gene correlation analysis**  
Finally, a preliminary examination of the correlation between gene values and response outcomes is performed. Genes with the strongest positive and negative correlations are saved for further analysis and feature selection. At this stage, it is clear that there are some features that behave distinctly with respect to drug response or lack thereof. The correlation was explored using both spearman and kendalls (which fit the case as they robust to outliers and non-linearity - the first which has not been taken into account as described in 1.j. and the other which should be assumed) and output the exact same results.
![Initial Gene Spearman Correlation](https://github.com/user-attachments/assets/4d9af6f5-96d3-405f-8d8e-3a01a7b896cb)

Nontheless, eventhough non-linerity was assumed, we can noticed some levels of correlation, giving good intuation on the relevant genes and shows a good predictable relations between the variables and the output.

Further, a pearson corrleation has also been validated, yeilding different results, which will be discussed in the _Model Training_ chapter.

**j. Outlier analysis**  
It is important to note that an outlier analysis was not conducted for two reasons:  

1. It is difficult to identify a statistical distribution in a ~60K-dimensional space, making it challenging to pinpoint statistically significant outliers across all dimensions.  
2. There are very few data points with known responses. Therefore, a point that might appear as an outlier could actually fall within the standard range — we simply lack enough data to confirm this.

A more detailed outlier analysis may be possible in the future with additional data collection.


### 2. FeatureExtractor

**a. Feature Extraction Methods**
Two primary methods were used for feature extraction: ANOVA (a filter method) and model-based methods.  
Wrapper methods (such as SFS or chi-test) were avoided due to their extreme slowness for data of this scale and the computational power limitations available.

**b. Train-Test Split**
Since some methods are model-based, the `FeatureExtractor` object performs an initial random split of the data into train-test sets, which are used to evaluate the model's performance with the selected features.

**c. Additional Methods**
In addition to ANOVA, two other methods were examined:  
- Logistic regression with L1 regularization (Lasso).  
- Random Forest (RF).  

Although XGBoost (XGB) could also be evaluated, it was largely excluded due to its significantly slower speed compared to RF when analyzing ~60K features. However, this can be configured differently if needed.

**d. Feature Selection**
From each method described in section 2.c., 2× `the desired number of features` (10, as specified in the exercise instructions, but configurable) are selected based on their highest weights in the model.  
The final set of features is chosen based on the highest agreement between the methods described in section c.2. and the highly correlated features identified during the EDA phase.

**e. Handling Disagreements**
If there is no agreement between the methods (Lasso, RF, ANOVA, and correlation) on the desired number of features, the remaining features are added based on the feature importance ranking in the Lasso method. This is because Lasso uniquely applies direct penalties (L1 penalty) to unimportant features, which is particularly important when dealing with ~60K features.

**f. PCA for Dimensionality Reduction**
Another feature extraction method explored involved dimensionality reduction using PCA.  
Two approaches can be applied for PCA:  
1. Defining the desired number of features in advance (less preferable).  
2. Defining the minimum variance that the eigenvectors should explain in the data.  

In this exercise, a requirement was set for at least **90% of the data variance** to be explained by the PCA features, which resulted in **30 extracted eigenvectors**.

**g. Rationale for Excluding Statistical Feature Aggregations**
Features such as the **mean/median response of genes per sample**, or the **maximum/minimum response per sample**, were not added.  
The reasoning for this decision is as follows:  
1. In the absence of prior biological knowledge or reliable online resources, it was assumed that some genes are entirely irrelevant to the final outcome.  
2. The potential for introducing significant noise into the results.  

As such, I did not find value in including these types of statistical aggregations. However, these could be investigated in future studies, given relevant biological research.

**h. Chosen Features**
The chosen features are as follows:
1. Using the _spearman_ correlation: `"226888_at", "210715_s_at", "216883_x_at", "1555656_at", "207958_at", "1569741_at", "244307_s_at", "235705_at", "212509_s_at", "222841_s_at"`
2. Using the _pearson_ correlation: `"1555656_at", "216883_x_at", "226888_at", "210715_s_at", "221566_s_at", "244307_s_at", "235705_at", "235695_at", "1569527_at", "222841_s_at"`
It is worth noting that 8 out of the 10 selected features exhibit high correlation in the results during the EDA stage, supporting the assumption of their relevance.

## 3. ModelTrainer - Model performance and interpretation.

### a. Selected Models
Since this is a classification problem, the following models were chosen:  
- Logistic Regression with L2 penalty: This was used to prevent over-reliance on a single parameter without silencing other features (avoiding sparsity after feature selection).  
- Random Forest (RF).  
- XGBoost (XGB).  
- A relatively basic fully connected neural network (NN) with 5 layers, including dropout layers to prevent overfitting and avoid unnecessary complexity.

### b. Selected Features
The features used were those described in section 2, for the reasons outlined there.

### c. Metrics for Evaluation
As this is a classification problem, and specifically a problem of drug responsiveness where sensitivity and specificity are of utmost importance, the metrics chosen to evaluate model performance were:  
- Overall accuracy  
- Sensitivity  
- Specificity  

### d. Training and Validation
Due to the limited dataset size, 5-split cross-validation was performed for evaluation. The performance results reflect the average values across the folds.

### e. Comparison: Original Features vs. PCA Features
As shown in the following graphs:  
![CV Average Performances Across Different Augmentations ](https://github.com/user-attachments/assets/360ceb98-903c-4711-80cf-ce32815d9ff3)

![CV Average Performances Across Different Augmentations (PCA)](https://github.com/user-attachments/assets/c8156fc9-2ff3-4923-ab4b-a4af8701e4c5)
 

It is evident that model performance on PCA-based features was significantly inferior compared to the "organic" features from the dataset. This trend persisted also when using features selected via Pearson correlation. Consequently, the PCA method was abandoned.

### f. Missing Values Study 
As demonstarted in the first plot in section 3.e. it is noticeable that additional data points are agumented, the performance of all models across all metrics declines. This behavior is consistent even when selecting features based on Pearson correlation, as shown in the following graph:  
![CV Average Performances Across Different Augmentations ](https://github.com/user-attachments/assets/20a50a4d-dc77-49d4-ba38-a366a5b965f1)

As a result, it was decided that the final model would only be trained on the original, reliable data ("organic"), without augmentation.

### g. Final Model Selection
Given the insights from the previous sections:  
- Organic features selected using a method incorporating Spearman correlation were chosen.  
- The final model selected was the neural network, as it demonstrated the highest average performance with the lowest variance compared to other models, despite the XGBoost model also showing impressive performance (hence it was retained as well).
- 37 training points vs 10 test points

![Model Performance Metrics (N=37 with y)](https://github.com/user-attachments/assets/21ecd99a-9ca0-463b-989c-8994bff2e04e)


Interestingly, when the models were trained on features extracted using Pearson correlation (which assumes linear relationships between variables and the outcome), there was a notable improvement in the performance of Logistic Regression, which encourages linear relationships. In contrast, there was a decline in the performance of decision tree-based models like RF and XGBoost - 

![Model Performance Metrics (N=37 with y)](https://github.com/user-attachments/assets/d9a12356-f5c2-4ece-889d-0846e0f609d8)

This trend can also be observed via the figures in 3.e. and 3.f.

These results align with logical expectations, highlighting the neural network's robustness to both linear and non-linear relationships, and the impact of linerity correlated chosen features.

### h. Final Model Performance
The final model's performance was evaluated on a one-time 80-20 (37 vs 10 data points) train-test split using organic features selected via Spearman correlation. The results are illustrated in the following graphs:  

- For the neural network:  
![NeuralNetwork_CM](https://github.com/user-attachments/assets/862a023e-5426-4b67-8731-8863d665171c)


- For XGBoost:  
![XGBoost_CM](https://github.com/user-attachments/assets/31e2500b-9cfc-400b-8993-664db18d2a44)



