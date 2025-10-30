# Machine Learning Project Report: Used Car Price Prediction

## I. Introduction and Problem Definition

This project addresses the problem of predicting the selling price of used cars based on the provided `train.csv` and `test.csv` datasets, which are likely associated with the "Cars 4 You: We buy your car!" project. The objective is to build a simple, yet effective, machine learning model to estimate the car's price, which can be used by management to understand pricing dynamics and inform business strategy.

The core problem is a **Regression** task, as the goal is to predict a continuous numerical variable: the car's price. The combined dataset contains **108,540 records** with features such as `Brand`, `model`, `year`, `mileage`, `engineSize`, and `transmission`. The initial data exploration revealed the presence of **missing values** and **inconsistencies** (e.g., negative mileage, misspellings in categorical features), necessitating a robust preprocessing strategy.

## II. Project Pipeline Schematic

The machine learning project followed a standard pipeline, broken down into five distinct stages:

| Stage | Techniques Used | Purpose |
| :--- | :--- | :--- |
| **1. Data Exploration** | Descriptive Statistics, Missing Value Analysis, Visualization (Histograms, Box Plots) | To understand data structure, identify key relationships, and detect data quality issues (missing values, inconsistencies). |
| **2. Preprocessing** | Imputation (Median/Mode), Feature Engineering (`Car_Age`), Log Transformation, Categorical Clean-up, One-Hot Encoding, Standard Scaling | To clean the data, handle missing values and inconsistencies, create meaningful features, handle skewness/outliers, and prepare data for modeling. |
| **3. Feature Selection** | Recursive Feature Elimination (RFE) | To select a powerful subset of 20 features from the high-dimensional feature space (138 features) that maximizes model performance and interpretability. |
| **4. Model Building** | Train-Test Split, Multiple Linear Regression | To train a simple predictive model on the selected features. |
| **5. Model Assessment** | R-squared (R2), Root Mean Squared Error (RMSE) | To evaluate the model's predictive accuracy and generalization ability on unseen data. |

## III. Data Preprocessing and Feature Selection Details

### A. Data Preprocessing (Task 2)

The preprocessing stage was critical for transforming the raw data into a format suitable for the Linear Regression model, with a focus on handling the newly identified data quality issues.

| Variable | Technique | Justification |
| :--- | :--- | :--- |
| **Missing Values** | **Imputation (Median/Mode)** | Numerical features (`mpg`, `tax`, etc.) were imputed with the **median**, and categorical features (`Brand`, `model`, etc.) with the **mode** to fill in gaps and preserve dataset size. |
| **price** | **Logarithmic Transformation** (`np.log1p`) | Applied to the target variable to handle its right-skewed distribution and mitigate the influence of outliers. |
| **year** | **Feature Engineering** (`Car_Age`) | Converted to `Car_Age` (2025 - year). Car age is a more intuitive and direct measure of depreciation. |
| **Inconsistencies** | **Clean-up & Correction** | The 'transmission' typo 'anual' was corrected to 'Manual', and negative 'mileage' values were corrected using the absolute value. |
| **Categorical Variables** | **Grouping & One-Hot Encoding** | Low-frequency categories in `Brand` and `model` were grouped into 'Other' to reduce dimensionality. Remaining categorical features were converted using **One-Hot Encoding** with `drop_first=True`. |
| **Numerical Features** | **Standard Scaling** | Applied to numerical features to standardize them (mean=0, std=1), which is essential for the Linear Regression model. |

### B. Feature Selection (Task 3)

Given the large number of features (138) after one-hot encoding, **Recursive Feature Elimination (RFE)** with a **Linear Regression** estimator was used to select a powerful subset of **20 features**.

**Justification for Final Selection:** The RFE approach provides a data-driven method for selecting the features that are collectively the most predictive for the Linear Regression model. The selected features include key factors such as `engineSize`, `mpg`, `Car_Age`, and specific one-hot encoded categories for `model` and `transmission`, which are the primary drivers of car price.

## IV. Model Building and Assessment (Task 4)

A **Multiple Linear Regression** model was chosen as the required "simple model" and trained on an 80/20 train-test split of the processed training data.

**Assessment Metrics and Results:**

| Metric | Training Set | Test Set |
| :--- | :---: | :---: |
| **R-squared (R2)** | 0.3511 | **0.3377** |
| **RMSE (Log Scale)** | 0.4295 | 0.4307 |
| **RMSE (Original Scale)** | N/A | **8046.60** |

The model achieved an **R-squared of 0.3377** on the test set, indicating that approximately 34% of the variance in the log-transformed price is explained by the model. The **Root Mean Squared Error (RMSE) on the original price scale is 8046.60**, which represents the average prediction error in the car's currency. The similar performance between the training and test sets confirms that the model is **well-generalized** and not overfitting.

The most significant features, as indicated by the model coefficients, are related to the **transmission type** (due to the presence of multiple, slightly different one-hot encoded columns for automatic/semi-automatic transmission) and key physical characteristics like `engineSize` and `mpg`.

***
*End of Report*
***
