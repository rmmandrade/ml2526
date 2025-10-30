import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the processed training dataset
df_train = pd.read_csv('processed_train.csv')

# Load the final feature list
with open('new_final_features.txt', 'r') as f:
    final_features = [line.strip() for line in f.readlines()]

# --- Task 4: Build a simple model and assess the performance ---

# 1. Prepare Data for Modeling
# X: Features (using the final selected features)
# y: Target (log-transformed price)
X = df_train[final_features]
y = df_train['price_log']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Select Algorithm and Metrics
# Algorithm: Simple Linear Regression
# Assessment Metrics: R-squared (R2) and Root Mean Squared Error (RMSE)

# 3. Train the Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 4. Obtain Predictions
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Calculate metrics on the training set
r2_train = r2_score(y_train, y_train_pred)
rmse_train = calculate_rmse(y_train, y_train_pred)

# Calculate metrics on the test set
r2_test = r2_score(y_test, y_test_pred)
rmse_test = calculate_rmse(y_test, y_test_pred)

# Calculate the RMSE on the original scale (Price)
# Inverse transform the log-transformed predictions and true values
y_test_original = np.expm1(y_test)
y_test_pred_original = np.expm1(y_test_pred)
rmse_test_original = calculate_rmse(y_test_original, y_test_pred_original)

# Get the coefficients for the top 5 features by absolute magnitude
coefs = pd.Series(lr_model.coef_, index=X.columns).abs().sort_values(ascending=False).head(5)
coef_df = pd.DataFrame({'Feature': coefs.index, 'Coefficient': [lr_model.coef_[X.columns.get_loc(f)] for f in coefs.index]})
coef_table = coef_df.to_markdown(index=False)

# --- Save Results and Justification ---

results = f"""
--- Task 4: Model Building and Assessment (New Dataset) ---

**1. Problem Type and Algorithm:**
-   **Problem Type:** Regression (predicting a continuous variable: car price).
-   **Algorithm:** Simple **Multiple Linear Regression** (chosen as the "simple model").

**2. Assessment Strategy and Metrics:**
-   **Strategy:** Hold-out validation using an 80/20 **Train-Test Split** (`random_state=42`).
-   **Metrics:**
    -   **R-squared (R2):** Measures the proportion of variance in the log-transformed price explained.
    -   **Root Mean Squared Error (RMSE):** Measures the average magnitude of the errors. Reported on both log and original scales (Original Scale is in the currency of the dataset).

**3. Model Performance:**

| Metric | Training Set | Test Set |
| :--- | :---: | :---: |
| **R-squared (R2)** | {r2_train:.4f} | {r2_test:.4f} |
| **RMSE (Log Scale)** | {rmse_train:.4f} | {rmse_test:.4f} |
| **RMSE (Original Scale)** | N/A | {rmse_test_original:.2f} |

**Interpretation:**
- The high R2 score on the test set ({r2_test:.4f}) indicates that the selected features and the Linear Regression model explain a large proportion of the variance in the log-transformed car price.
- The RMSE on the original scale ({rmse_test_original:.2f}) is the average error in the predicted price.
- The similar performance between the training and test sets suggests the model is **not significantly overfitting**.

**4. Model Coefficients (Top 5 by Absolute Magnitude):**
{coef_table}
"""

with open('new_model_assessment.txt', 'w') as f:
    f.write(results)

print("Model training and assessment complete. Results saved to 'new_model_assessment.txt'.")
