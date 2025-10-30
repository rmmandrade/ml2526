import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset
df = pd.read_csv('car_data.csv')

# --- Task 2: Clean and pre-process the dataset (5 points) ---

# 1. Feature Creation: Create 'Car_Age' feature
# Assuming current year is 2025 based on the handout date (September, 2025)
df['Car_Age'] = 2025 - df['Year']
df.drop('Year', axis=1, inplace=True) # Drop the original 'Year' column

# 2. Outlier Handling: Log-transform 'Present_Price' and 'Selling_Price'
# This addresses the high right-skewness and the presence of outliers, making the distribution more Gaussian-like for linear models.
df['Selling_Price_Log'] = np.log1p(df['Selling_Price'])
df['Present_Price_Log'] = np.log1p(df['Present_Price'])
df.drop(['Selling_Price', 'Present_Price'], axis=1, inplace=True) # Drop original columns

# 3. Categorical Variables: Deal with 'Car_Name'
# Extract the brand name from 'Car_Name' as a new feature
df['Brand'] = df['Car_Name'].apply(lambda x: x.split(' ')[0])
df.drop('Car_Name', axis=1, inplace=True) # Drop the original 'Car_Name' column

# Check for low-frequency brands and group them as 'Other'
brand_counts = df['Brand'].value_counts()
rare_brands = brand_counts[brand_counts < 10].index
df['Brand'] = df['Brand'].replace(rare_brands, 'Other')

# 4. Categorical Variables: One-Hot Encoding
# Select categorical columns for encoding
categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Brand']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 5. Feature Creation: Create a feature for 'No_of_Owners' as categorical
# This is already numerical (0, 1, 3), but we will keep it as is for now, as it represents discrete categories.
# The 'Owner' column is already good to use as is, but we will rename it to be more descriptive.
df.rename(columns={'Owner': 'No_of_Owners'}, inplace=True)


# 6. Data Scaling: Scale numerical features
# Identify numerical features that are not the target variable
numerical_cols_to_scale = ['Kms_Driven', 'Car_Age', 'No_of_Owners', 'Present_Price_Log']

# Use StandardScaler for features that are approximately Gaussian after log-transformation
scaler = StandardScaler()
df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])

print("--- Data Preprocessing Summary (Task 2) ---")
print("\nShape of the processed dataset:", df.shape)
print("\nFirst 5 rows of the processed dataset:")
print(df.head())
print("\nData types:")
df.info()

# Save the processed dataset for the next steps
df.to_csv('processed_car_data.csv', index=False)

# Save the reasoning for the report
reasoning = """
--- Task 2: Data Cleaning and Pre-processing Reasoning ---

1. Missing Values:
   - The initial data exploration confirmed there are no missing values in the dataset (301 non-null entries out of 301 rows). No action was required.

2. Outlier Handling:
   - **Action:** The target variable ('Selling_Price') and the highly correlated feature ('Present_Price') were highly right-skewed, indicating the presence of high-value outliers. To mitigate the influence of these outliers and make the distributions more normal, a **logarithmic transformation (np.log1p)** was applied. The new target variable is 'Selling_Price_Log'.

3. Categorical Variables:
   - **Car_Name:** The full car name is too granular. The **brand name** was extracted as a new feature ('Brand'). Brands with a count of less than 10 were grouped into an 'Other' category to reduce the dimensionality of the feature space and prevent overfitting.
   - **One-Hot Encoding:** Categorical features ('Fuel_Type', 'Seller_Type', 'Transmission', 'Brand') were converted into numerical format using **One-Hot Encoding** with `drop_first=True` to avoid multicollinearity (the dummy variable trap).

4. Feature Creation:
   - **Car_Age:** The 'Year' column was converted into a more meaningful feature, 'Car_Age' (2025 - Year), as the age of a car is a direct factor in its depreciation. The original 'Year' column was dropped.

5. Data Scaling:
   - **Action:** Numerical features ('Kms_Driven', 'Car_Age', 'No_of_Owners', 'Present_Price_Log') were scaled using **StandardScaler**. This technique standardizes the features by removing the mean and scaling to unit variance, which is crucial for distance-based and gradient-descent-based models to converge faster and perform better. The target variable 'Selling_Price_Log' was intentionally *not* scaled, as is common practice in regression tasks.
"""
with open('preprocessing_reasoning.txt', 'w') as f:
    f.write(reasoning)

print("\nProcessed data saved to 'processed_car_data.csv'.")
print("Preprocessing reasoning saved to 'preprocessing_reasoning.txt'.")
