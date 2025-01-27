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

### 1. DataHandler

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

