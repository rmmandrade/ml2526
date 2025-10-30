# -*- coding: utf-8 -*-
"""
# Machine Learning Project: Used Car Price Prediction
## Homework_GroupAI

This notebook covers the machine learning project on predicting used car prices, following the tasks outlined in the project handout and using the provided `train.csv` and `test.csv` datasets.

**Dataset:** Provided `train.csv` and `test.csv` (Kaggle Competition Data)
**Problem Type:** Regression
"""

# %% [markdown]
"""
## I. Import Libraries and Load Data (Task 1, Part 1)
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer

# Define the target and ID columns
TARGET_COL = 'price'
ID_COL = 'carID'

# Load the datasets
train_df = pd.read_csv('/home/ubuntu/upload/train.csv')
test_df = pd.read_csv('/home/ubuntu/upload/test.csv')

# Add a source column to identify the origin of the data
train_df['Source'] = 'train'
test_df['Source'] = 'test'

# Store test IDs for submission file
test_ids = test_df[ID_COL]
test_df['ID_for_submission'] = test_ids

# Drop target from test set if present and align columns
if TARGET_COL in test_df.columns:
    test_df.drop(TARGET_COL, axis=1, inplace=True)

# Concatenate the datasets for consistent feature engineering and preprocessing
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# %% [markdown]
"""
## II. Data Exploration and Analysis (Task 1: 3 points)

### 1. Check Data Contents and Descriptive Statistics

Initial inspection reveals a large dataset with missing values and some inconsistencies.
"""

# %%
print("--- Initial Data Exploration ---")
print(f"Shape of the combined dataset: {combined_df.shape}")
print("\nData types and non-null counts:")
combined_df.info()
print("\nMissing values count per column:")
print(combined_df.isnull().sum())
print("\nDescriptive statistics for numerical columns:")
print(combined_df.describe())

# %% [markdown]
"""
### 2. Visual Exploration and Insights

A new feature, `Car_Age`, is created for better analysis. Visualizations are used to understand the target variable's distribution and its relationship with key features.
"""

# %%
# Create 'Car_Age' feature for visualization
combined_df['Car_Age'] = 2025 - combined_df['year']

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Distribution of the target variable (price)
plt.figure(figsize=(10, 6))
sns.histplot(combined_df['price'].dropna(), kde=True, bins=50)
plt.title('Distribution of Car Price')
plt.xlabel('Price')
plt.show()

# Price vs. Car Age
plt.figure(figsize=(10, 6))
sns.regplot(x='Car_Age', y='price', data=combined_df, scatter_kws={'alpha':0.3})
plt.title('Price vs. Car Age')
plt.xlabel('Car Age (Years)')
plt.ylabel('Price')
plt.show()

# Categorical features vs. Price
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.boxplot(x='transmission', y='price', data=combined_df, ax=axes[0])
axes[0].set_title('Price by Transmission Type')
sns.boxplot(x='fuelType', y='price', data=combined_df, ax=axes[1])
axes[1].set_title('Price by Fuel Type')
sns.boxplot(x='hasDamage', y='price', data=combined_df, ax=axes[2])
axes[2].set_title('Price by Damage Status')
plt.tight_layout()
plt.show()

# %% [markdown]
"""
**Insights from Exploration (Rationale and Findings):**

1.  **Data Quality:** Significant missing values are present across most features, requiring imputation.
2.  **Target Skewness:** The `price` is highly right-skewed, necessitating log transformation.
3.  **Inconsistencies:** Typographical errors (e.g., 'anual' in transmission) and negative values (e.g., in `mileage`) must be corrected.
4.  **Key Predictors:** Price shows a clear negative correlation with `Car_Age` and `mileage`, and strong differentiation across `transmission` and `fuelType`.
"""

# %% [markdown]
"""
## III. Clean and Pre-process the Dataset (Task 2: 5 points)

This section details the steps taken to clean and transform the data, including imputation, feature engineering, and encoding.
"""

# %%
df_proc = combined_df.copy()

# 1. Feature Creation: Car_Age and Log Transformation
df_proc.drop('year', axis=1, inplace=True)
df_proc['price_log'] = np.log1p(df_proc['price'])
df_proc.drop('price', axis=1, inplace=True)

# 2. Handling Missing Values (Imputation)
numerical_cols = ['mileage', 'engineSize', 'tax', 'mpg', 'paintQuality%', 'previousOwners', 'Car_Age', 'hasDamage']
categorical_cols = ['Brand', 'model', 'transmission', 'fuelType']

num_imputer = SimpleImputer(strategy='median')
df_proc[numerical_cols] = num_imputer.fit_transform(df_proc[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df_proc[categorical_cols] = cat_imputer.fit_transform(df_proc[categorical_cols])

# 3. Handling Inconsistencies and Categorical Clean-up
df_proc['transmission'] = df_proc['transmission'].replace('anual', 'Manual')
df_proc['mileage'] = df_proc['mileage'].abs() # Correct negative mileage

# Group low-frequency 'Brand' and 'model' into 'Other'
brand_counts = df_proc['Brand'].value_counts()
rare_brands = brand_counts[brand_counts < 1000].index
df_proc['Brand'] = df_proc['Brand'].replace(rare_brands, 'Other_Brand')

model_counts = df_proc['model'].value_counts()
rare_models = model_counts[model_counts < 500].index
df_proc['model'] = df_proc['model'].replace(rare_models, 'Other_Model')

# 4. One-Hot Encoding
categorical_cols_to_encode = ['Brand', 'model', 'transmission', 'fuelType']
df_proc = pd.get_dummies(df_proc, columns=categorical_cols_to_encode, drop_first=True)

# 5. Data Scaling (StandardScaler)
numerical_cols_to_scale = ['mileage', 'engineSize', 'tax', 'mpg', 'paintQuality%', 'previousOwners', 'Car_Age']
scaler = StandardScaler()
df_proc[numerical_cols_to_scale] = scaler.fit_transform(df_proc[numerical_cols_to_scale])

# Separate back into train and test sets
df_train = df_proc[df_proc['Source'] == 'train'].drop('Source', axis=1)
df_test = df_proc[df_proc['Source'] == 'test'].drop(['Source', 'price_log'], axis=1)

# Drop ID columns not needed for modeling
df_train.drop([ID_COL, 'ID_for_submission'], axis=1, inplace=True)
df_test.drop([ID_COL, 'ID_for_submission'], axis=1, inplace=True)

# Save processed data for persistence (optional in a notebook, but good practice)
df_train.to_csv('processed_train.csv', index=False)
df_test.to_csv('processed_test.csv', index=False)

print("Processed training data shape:", df_train.shape)
print("Processed test data shape:", df_test.shape)

# %% [markdown]
"""
## IV. Feature Selection (Task 3: 3 points)

A clear and unambiguous strategy is defined and implemented using **Recursive Feature Elimination (RFE)** with a Linear Regression estimator to select the top 20 features from the high-dimensional feature space.
"""

# %%
# Separate features (X) and target (y)
X = df_train.drop(columns=['price_log'])
y = df_train['price_log']

# RFE to select a reasonable number of features (e.g., 20)
estimator = LinearRegression()
n_features_to_select = 20

rfe = RFE(estimator, n_features_to_select=n_features_to_select)
rfe.fit(X, y)

# Get the selected features
rfe_support = pd.Series(rfe.support_, index=X.columns)
final_features = list(X.columns[rfe_support])

print("--- Feature Selection Results ---")
print(f"Number of features selected: {len(final_features)}")
print("Final Selected Features:")
for feature in final_features:
    print(f"- {feature}")

# %% [markdown]
"""
## V. Build a Simple Model and Assess Performance (Task 4: 4 points)

### 1. Model Selection and Assessment Strategy

-   **Algorithm:** Simple **Multiple Linear Regression**.
-   **Assessment Strategy:** 80/20 **Train-Test Split** on the training data.
-   **Metrics:** **R-squared (R2)** and **Root Mean Squared Error (RMSE)**.
"""

# %%
# Prepare Data for Modeling
X_model = df_train[final_features]
y_model = df_train['price_log']

# Split the data
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_model, y_model, test_size=0.2, random_state=42)

# Train the Model
lr_model = LinearRegression()
lr_model.fit(X_train_split, y_train_split)

# Obtain Predictions
y_test_pred_split = lr_model.predict(X_test_split)

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Assess Performance
r2_test = r2_score(y_test_split, y_test_pred_split)

# RMSE on the original scale (Price)
y_test_original = np.expm1(y_test_split)
y_test_pred_original = np.expm1(y_test_pred_split)
rmse_test_original = calculate_rmse(y_test_original, y_test_pred_original)

print("--- Model Performance Assessment (Test Split) ---")
print(f"R-squared (Test Set): {r2_test:.4f}")
print(f"RMSE (Original Scale): {rmse_test_original:.2f}")

# %% [markdown]
"""
## VI. Generate Predictions for Submission

The final model is trained on the entire processed training set and used to generate predictions for the provided `test.csv` data.
"""

# %%
# Train the final model on the entire processed training set
lr_model_final = LinearRegression()
lr_model_final.fit(X, y)

# Prepare Test Data for Prediction
missing_cols = set(final_features) - set(df_test.columns)
for c in missing_cols:
    df_test[c] = 0

# Select and align the features in the test set
X_test_final = df_test[final_features]

# Generate Predictions
y_test_pred_log = lr_model_final.predict(X_test_final)

# Inverse transform the log-transformed predictions to get the final price
y_test_pred = np.expm1(y_test_pred_log)

# Create Submission File
submission_df = pd.DataFrame({
    'carID': test_ids.values,
    'price': y_test_pred
})

# Round the price to 2 decimal places
submission_df['price'] = submission_df['price'].round(2)

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")
print("First 5 predictions:")
print(submission_df.head())

# %% [markdown]
"""
## VII. Conclusion and Deliverables

The project successfully implemented a machine learning pipeline for used car price prediction. The final deliverables are:

1.  This Jupyter Notebook (`Homework_GroupAI.ipynb`) containing all code and markdown comments.
2.  A 2-page PDF report (`Project_Pipeline_Report.pdf`) describing the overall structure and rationale.
3.  The final prediction file (`submission.csv`) for the test set.
"""
