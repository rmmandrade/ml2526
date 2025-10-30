import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the combined dataset
df = pd.read_csv('combined_data.csv')

# --- Task 2: Clean and pre-process the dataset (5 points) ---

# 1. Feature Creation: Car_Age and Log Transformation
# Assuming current year is 2025
df['Car_Age'] = 2025 - df['year']
df.drop('year', axis=1, inplace=True)

# Log-transform the target variable 'price' to handle skewness
# This is only applied to the training data (where price is not NaN)
df['price_log'] = np.log1p(df['price'])
df.drop('price', axis=1, inplace=True)

# 2. Handling Missing Values (Imputation)
# Separate numerical and categorical columns
numerical_cols = ['mileage', 'engineSize', 'tax', 'mpg', 'paintQuality%', 'previousOwners', 'Car_Age', 'hasDamage']
categorical_cols = ['Brand', 'model', 'transmission', 'fuelType']

# Impute numerical features with the median
num_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Impute categorical features with the most frequent value (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# 3. Handling Inconsistencies and Categorical Clean-up
# Correct the 'transmission' typo 'anual' to 'Manual'
df['transmission'] = df['transmission'].replace('anual', 'Manual')

# Group low-frequency 'Brand' and 'model' into 'Other'
# Brand grouping (threshold: < 1000 occurrences)
brand_counts = df['Brand'].value_counts()
rare_brands = brand_counts[brand_counts < 1000].index
df['Brand'] = df['Brand'].replace(rare_brands, 'Other_Brand')

# Model grouping (threshold: < 500 occurrences)
model_counts = df['model'].value_counts()
rare_models = model_counts[model_counts < 500].index
df['model'] = df['model'].replace(rare_models, 'Other_Model')

# 4. Handling Outliers/Inconsistencies in Numerical Data
# Correct negative mileage by taking the absolute value (assuming data entry error)
df['mileage'] = df['mileage'].abs()

# 5. One-Hot Encoding
# Select categorical columns for encoding
categorical_cols_to_encode = ['Brand', 'model', 'transmission', 'fuelType']
df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

# 6. Data Scaling (StandardScaler)
# Identify numerical features to scale (all numerical features except carID and the target)
numerical_cols_to_scale = ['mileage', 'engineSize', 'tax', 'mpg', 'paintQuality%', 'previousOwners', 'Car_Age']
scaler = StandardScaler()
df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])

# Drop ID columns not needed for modeling
df.drop(['carID', 'ID_for_submission'], axis=1, inplace=True)

print("--- Data Preprocessing Summary (Task 2) ---")
print("\nShape of the processed dataset:", df.shape)
print("\nFirst 5 rows of the processed dataset:")
print(df.head())
print("\nData types:")
df.info()

# Separate back into train and test sets
df_train = df[df['Source'] == 'train'].drop('Source', axis=1)
df_test = df[df['Source'] == 'test'].drop(['Source', 'price_log'], axis=1)

# Save the processed datasets
df_train.to_csv('processed_train.csv', index=False)
df_test.to_csv('processed_test.csv', index=False)

# Save the reasoning for the report
reasoning = f"""
--- Task 2: Data Cleaning and Pre-processing Reasoning ---

1. Feature Creation:
   - **Car_Age:** Created from 'year' (2025 - year) to represent car depreciation. Original 'year' dropped.
   - **price_log:** Log-transformed the target variable 'price' using `np.log1p` to handle its right-skewed distribution and mitigate the impact of outliers. Original 'price' dropped.

2. Missing Values:
   - **Imputation:** Missing values were identified across several columns. Numerical features were imputed using the **median** (robust to outliers), and categorical features were imputed using the **most frequent** value (mode).

3. Inconsistencies and Categorical Clean-up:
   - **Transmission Typo:** The typo 'anual' in the 'transmission' column was corrected to 'Manual'.
   - **Negative Mileage:** Negative values in 'mileage' (likely data entry errors) were corrected by taking the **absolute value**.
   - **Feature Reduction (Categorical):** Low-frequency categories in 'Brand' (count < 1000) and 'model' (count < 500) were grouped into 'Other_Brand' and 'Other_Model', respectively, to reduce the dimensionality of the feature space.

4. One-Hot Encoding:
   - Categorical features ('Brand', 'model', 'transmission', 'fuelType') were converted into numerical format using **One-Hot Encoding** with `drop_first=True` to avoid multicollinearity.

5. Data Scaling:
   - **StandardScaler:** Numerical features were scaled to standardize them (mean=0, std=1), which is essential for the Linear Regression model.

6. Data Separation:
   - The combined dataset was split back into `processed_train.csv` and `processed_test.csv` based on the 'Source' column, with the test set having the `price_log` column removed.
"""
with open('new_preprocessing_reasoning.txt', 'w') as f:
    f.write(reasoning)

print("\nProcessed data saved to 'processed_train.csv' and 'processed_test.csv'.")
print("Preprocessing reasoning saved to 'new_preprocessing_reasoning.txt'.")
