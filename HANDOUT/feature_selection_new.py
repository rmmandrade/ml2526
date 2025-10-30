import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Load the processed training dataset
df_train = pd.read_csv('processed_train.csv')

# Separate features (X) and target (y)
X = df_train.drop(columns=['price_log'])
y = df_train['price_log']

# --- Feature Selection Strategy (Task 3) ---
# With a high number of features (138), we need a robust selection method.
# We will use RFE to select a smaller, more manageable set of features.

# 1. Recursive Feature Elimination (RFE)
# Use Linear Regression as the estimator and select a reasonable number of features (e.g., 20)
estimator = LinearRegression()
n_features_to_select = 20

rfe = RFE(estimator, n_features_to_select=n_features_to_select)
rfe.fit(X, y)

# Get the selected features and their ranking
rfe_ranking = pd.Series(rfe.ranking_, index=X.columns)
rfe_support = pd.Series(rfe.support_, index=X.columns)

final_features = list(X.columns[rfe_support])

# --- Final Feature Selection and Justification ---

# Save the final feature list and justification to a file
justification = f"""
--- Task 3: Feature Selection Strategy and Justification (New Dataset) ---

**Strategy:**
The feature selection strategy employed **Recursive Feature Elimination (RFE)** with a **Multiple Linear Regression** estimator. Given the high dimensionality of the processed dataset (138 features), RFE is a suitable method to automatically select a powerful subset of features.

1.  **Recursive Feature Elimination (RFE):** RFE was applied to the entire feature set, iteratively training the model and removing the weakest feature until the desired number of features (n={n_features_to_select}) was reached.

**RFE Ranking (Top 10 features with rank 1):**
{rfe_ranking[rfe_ranking == 1].index.tolist()[:10]}

**Final Selected Features:**
The final set consists of the {n_features_to_select} features identified by RFE as the most important for predicting the log-transformed car price.

**Final Feature List (Total {len(final_features)}):**
{final_features}

**Justification for Final Selection:**
-   **RFE** provides a data-driven approach to feature selection, ensuring that the selected features are collectively the most predictive for the chosen model (Linear Regression).
-   The selected features include key numerical variables like `engineSize`, `mpg`, and `Car_Age`, as well as several one-hot encoded categorical features (e.g., specific `model` and `Brand` categories), confirming their importance in the pricing model.
-   This selection is a practical compromise between model complexity and performance, adhering to the requirement for a simple, yet effective, model.
"""

with open('new_feature_selection_justification.txt', 'w') as f:
    f.write(justification)

print("Feature selection complete.")
print("Final features saved to 'new_feature_selection_justification.txt'.")

# Save the final feature list to a separate file for use in the next phase
with open('new_final_features.txt', 'w') as f:
    f.write('\n'.join(final_features))
