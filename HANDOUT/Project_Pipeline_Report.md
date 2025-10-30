# Machine Learning Project Report: Used Car Price Prediction

## I. Introduction and Problem Definition

This project addresses the problem of predicting the selling price of used cars based on a dataset likely derived from a competition such as "Cars 4 You: We buy your car!". The objective is to build a simple, yet effective, machine learning model to estimate the car's price, which can be used by management to understand pricing dynamics and inform business strategy.

The core problem is a **Regression** task, as the goal is to predict a continuous numerical variable: the car's selling price (in Lakhs). The initial data exploration revealed **301 records** with **9 features**, including `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, and `Owner`. The dataset was found to be clean, with no missing values, but the target variable (`Selling_Price`) was highly right-skewed, indicating the presence of high-value outliers.

## II. Project Pipeline Schematic

The machine learning project followed a standard pipeline, broken down into five distinct stages:

| Stage | Techniques Used | Purpose |
| :--- | :--- | :--- |
| **1. Data Exploration** | Descriptive Statistics, Correlation Analysis, Visualization (Histograms, Scatter Plots, Box Plots) | To understand data structure, identify key relationships, and detect anomalies. |
| **2. Preprocessing** | Feature Engineering (`Car_Age`), Log Transformation, One-Hot Encoding, Standard Scaling | To clean the data, create meaningful features, handle skewness/outliers, and prepare data for modeling. |
| **3. Feature Selection** | Domain Knowledge, Correlation, Recursive Feature Elimination (RFE) | To select an optimal subset of features that maximizes model performance and interpretability. |
| **4. Model Building** | Train-Test Split, Multiple Linear Regression | To train a simple predictive model on the selected features. |
| **5. Model Assessment** | R-squared (R2), Root Mean Squared Error (RMSE) | To evaluate the model's predictive accuracy and generalization ability on unseen data. |

## III. Data Preprocessing and Feature Selection Details

### A. Data Preprocessing (Task 2)

The preprocessing stage was critical for transforming the raw data into a format suitable for the Linear Regression model.

| Variable | Technique | Justification |
| :--- | :--- | :--- |
| **Selling\_Price, Present\_Price** | **Logarithmic Transformation** (`np.log1p`) | The prices were highly right-skewed. Log transformation made the distribution more normal, mitigating the influence of high-value outliers and improving the linear model's performance. |
| **Year** | **Feature Engineering** (`Car_Age`) | Converted to `Car_Age` (2025 - Year). Car age is a more intuitive and direct measure of depreciation than the year of manufacture. |
| **Car\_Name** | **Feature Engineering** (`Brand`) | Extracted the car brand and dropped the granular model name. Low-frequency brands were grouped into an 'Other' category to reduce the number of features and prevent overfitting. |
| **Categorical Variables** | **One-Hot Encoding** | Converted nominal categorical features (`Fuel_Type`, `Seller_Type`, `Transmission`, `Brand`) into a numerical format (binary columns) using `drop_first=True` to avoid multicollinearity. |
| **Numerical Features** | **Standard Scaling** | Scaled features (`Kms_Driven`, `Car_Age`, `No_of_Owners`, `Present_Price_Log`) to have a mean of 0 and a standard deviation of 1. This ensures all features contribute equally to the model training process and accelerates convergence. |

### B. Feature Selection (Task 3)

The feature selection aimed to identify the most significant predictors for the model, using a hybrid approach:

1.  **Domain Knowledge/Correlation:** The log-transformed **Present Price** (`Present_Price_Log`) was immediately retained due to its exceptionally high correlation with the target variable (R-squared of 0.88).
2.  **Recursive Feature Elimination (RFE):** RFE with a **Linear Regression** estimator was applied to the remaining features to select the top 9 most important ones.

The final selected feature set, which includes `Present_Price_Log` and the top 9 RFE-selected features, is:

*   `Present_Price_Log`
*   `Fuel_Type_Diesel`, `Fuel_Type_Petrol`
*   `Seller_Type_Individual`
*   `Transmission_Manual`
*   `Brand_Royal`, `Brand_brio`, `Brand_city`, `Brand_corolla`, `Brand_fortuner`, `Brand_verna`

**Justification:** This selection captures the most influential factors identified in the exploration phase: the car's current market value (`Present_Price_Log`), the type of seller, the transmission, the fuel type, and the brand's premium/non-premium status.

## IV. Model Building and Assessment (Task 4)

A **Multiple Linear Regression** model was chosen as the required "simple model" and trained on an 80/20 train-test split.

**Assessment Metrics and Results:**

| Metric | Training Set | Test Set |
| :--- | :---: | :---: |
| **R-squared (R2)** | 0.9188 | **0.8757** |
| **RMSE (Log Scale)** | 0.2305 | 0.2755 |
| **RMSE (Original Scale - Lakhs)** | N/A | **2.2033** |

The model performed strongly, achieving an **R-squared of 0.8757** on the test set, meaning over 87% of the variance in the log-transformed selling price is explained by the model. The **Root Mean Squared Error (RMSE) on the original price scale is 2.20 Lakhs**, indicating that the model's average prediction error is approximately ₹2.20 Lakhs. The similar performance between the training and test sets confirms that the model is **well-generalized** and not overfitting.

The model coefficients confirmed that `Present_Price_Log` is the dominant predictor, while being an `Individual` seller or having a `Manual` transmission significantly **reduces** the predicted selling price compared to the baseline (Dealer, Automatic).

***
*End of Report*
***
