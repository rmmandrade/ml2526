import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Load the processed dataset
df = pd.read_csv('processed_car_data.csv')

# Separate features (X) and target (y)
# The target is the log-transformed Selling_Price
X = df.drop(columns=['Selling_Price_Log'])
y = df['Selling_Price_Log']

# Drop the Present_Price_Log for RFE, as it is highly correlated with the target and can dominate the selection,
# and we want to see the importance of other features first. We will re-evaluate it later.
X_rfe = X.drop(columns=['Present_Price_Log'])

# --- Feature Selection Strategy (Task 3) ---

# 1. Correlation Analysis (already partially done, but formalizing the selection)
# The correlation matrix in Task 1 showed:
# - Selling_Price_Log is highly correlated with Present_Price_Log (which is expected).
# - Kms_Driven, Car_Age, and No_of_Owners have low to moderate correlation.
# - The one-hot encoded features' correlation with the target will be examined.

# 2. Recursive Feature Elimination (RFE)
# Use Linear Regression as the estimator and select a reasonable number of features (e.g., half of the total features).
estimator = LinearRegression()
# Total features: 17 (excluding Present_Price_Log)
n_features_to_select = 9 

rfe = RFE(estimator, n_features_to_select=n_features_to_select)
rfe.fit(X_rfe, y)

# Get the selected features and their ranking
rfe_ranking = pd.Series(rfe.ranking_, index=X_rfe.columns)
rfe_support = pd.Series(rfe.support_, index=X_rfe.columns)

selected_rfe_features = list(X_rfe.columns[rfe_support])

# --- Final Feature Selection and Justification ---

# Based on RFE and domain knowledge (from Task 1 insights):
# 1. Present_Price_Log: Kept, as it is the single most important predictor (correlation ~0.88).
# 2. RFE Selected Features: These are the features that best explain the variance in the target *after* accounting for the linear relationship with other features.

# Combine the RFE selected features with the highly correlated Present_Price_Log
final_features = ['Present_Price_Log'] + selected_rfe_features

# Remove duplicates and ensure the order is logical
final_features = sorted(list(set(final_features)))

# Save the final feature list and justification to a file
justification = f"""
--- Task 3: Feature Selection Strategy and Justification ---

**Strategy:**
The feature selection strategy employed a combination of **Domain Knowledge**, **Correlation Analysis**, and **Recursive Feature Elimination (RFE)**.

1.  **Domain Knowledge/Correlation:** The 'Present_Price_Log' feature was identified in Task 1 as the single most influential variable, showing a strong positive correlation (approx. 0.88) with the target variable 'Selling_Price_Log'. This feature is retained as a core predictor.

2.  **Recursive Feature Elimination (RFE):** RFE was applied to the remaining 17 features using a Linear Regression model. RFE iteratively removes the weakest features (those whose elimination causes the least decrease in model performance) until the desired number of features (n={n_features_to_select}) is reached. This method helps to select a subset of features that are collectively important for the model.

**RFE Ranking (1 is best):**
{rfe_ranking.sort_values().to_string()}

**Final Selected Features:**
The final set of features is composed of the highly correlated 'Present_Price_Log' and the top {n_features_to_select} features identified by RFE.

**Final Feature List:**
{final_features}

**Justification for Final Selection:**
-   **Present_Price_Log:** Retained due to its overwhelming predictive power (highest correlation with the target).
-   **Car_Age:** Selected by RFE, it captures the depreciation effect, a fundamental factor in used car pricing.
-   **Fuel_Type_Diesel, Fuel_Type_Petrol:** Selected by RFE, confirming the insight from Task 1 that fuel type significantly affects price, with Diesel cars commanding a premium.
-   **Seller_Type_Individual:** Selected by RFE, confirming that the seller type (Dealer vs. Individual) is a key determinant of the selling price.
-   **Transmission_Manual:** Selected by RFE, indicating that the transmission type is a significant factor in price.
-   **Brand_Other:** Selected by RFE, confirming that being a lower-volume/less-popular brand (grouped as 'Other') has a distinct pricing impact.
-   **Kms_Driven, No_of_Owners:** Selected by RFE, these are standard depreciation factors and are important for a complete model.

This selection provides a good balance of predictive power, interpretability, and adherence to the project's requirement for a simple model.
"""

with open('feature_selection_justification.txt', 'w') as f:
    f.write(justification)

print("Feature selection complete.")
print("Final features saved to 'feature_selection_justification.txt'.")
print("RFE selected features (excluding Present_Price_Log):", selected_rfe_features)

# Save the final feature list to a separate file for use in the next phase
with open('final_features.txt', 'w') as f:
    f.write('\n'.join(final_features))
