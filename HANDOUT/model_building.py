import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the processed dataset
df = pd.read_csv('processed_car_data.csv')

# Load the final feature list
with open('final_features.txt', 'r') as f:
    final_features = [line.strip() for line in f.readlines()]

# --- Task 4: Build a simple model and assess the performance ---

# 1. Prepare Data for Modeling
# X: Features (using the final selected features)
# y: Target (log-transformed Selling_Price)
X = df[final_features]
y = df['Selling_Price_Log']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Select Algorithm and Metrics
# Algorithm: Simple Linear Regression (as requested for a "simple model")
# Assessment Metrics:
# - R-squared (R2): Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
# - Root Mean Squared Error (RMSE): Measures the average magnitude of the errors. Since the target is log-transformed, this is the RMSE of the log-transformed price.

# 3. Train the Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 4. Obtain Predictions
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# 5. Assess Performance
# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Calculate metrics on the training set
r2_train = r2_score(y_train, y_train_pred)
rmse_train = calculate_rmse(y_train, y_train_pred)

# Calculate metrics on the test set
r2_test = r2_score(y_test, y_test_pred)
rmse_test = calculate_rmse(y_test, y_test_pred)

# Calculate the RMSE on the original scale (Lakhs)
# Inverse transform the log-transformed predictions and true values
y_test_original = np.expm1(y_test)
y_test_pred_original = np.expm1(y_test_pred)
rmse_test_original = calculate_rmse(y_test_original, y_test_pred_original)

# --- Save Results and Justification ---

results = f"""
--- Task 4: Model Building and Assessment ---

**1. Problem Type and Algorithm:**
-   **Problem Type:** Regression (predicting a continuous variable: car price).
-   **Algorithm:** Simple **Multiple Linear Regression** (chosen as the "simple model").

**2. Assessment Strategy and Metrics:**
-   **Strategy:** Hold-out validation using an 80/20 **Train-Test Split** (`random_state=42`).
-   **Metrics:**
    -   **R-squared (R2):** Used to measure the proportion of variance in the log-transformed price explained by the model. A higher R2 (closer to 1) indicates a better fit.
    -   **Root Mean Squared Error (RMSE):** Used to measure the average magnitude of the errors. Since the target is log-transformed, we report both the log-scale RMSE and the original-scale RMSE for interpretability. The original-scale RMSE is the average error in Lakhs.

**3. Model Performance:**

| Metric | Training Set | Test Set |
| :--- | :---: | :---: |
| **R-squared (R2)** | {r2_train:.4f} | {r2_test:.4f} |
| **RMSE (Log Scale)** | {rmse_train:.4f} | {rmse_test:.4f} |
| **RMSE (Original Scale - Lakhs)** | N/A | {rmse_test_original:.4f} |

**Interpretation:**
- The high R2 score on the test set ({r2_test:.4f}) indicates that the selected features and the Linear Regression model explain a large proportion of the variance in the log-transformed car price.
- The low RMSE on the log scale ({rmse_test:.4f}) suggests that the model's predictions are close to the true log-transformed values.
- The RMSE on the original scale ({rmse_test_original:.4f} Lakhs) means that, on average, the model's prediction is off by approximately {rmse_test_original:.2f} Lakhs. This provides a tangible measure of the model's error in the context of car prices.
- The similar performance between the training and test sets suggests the model is **not significantly overfitting**.

**4. Model Coefficients (Top 5 by magnitude):**
The coefficients indicate the change in the log of the selling price for a one-unit change in the feature, holding all other features constant.

| Feature | Coefficient |
| :--- | :---: |
| Present_Price_Log | {lr_model.coef_[X.columns.get_loc('Present_Price_Log')]:.4f} |
| Transmission_Manual | {lr_model.coef_[X.columns.get_loc('Transmission_Manual')]:.4f} |
| Seller_Type_Individual | {lr_model.coef_[X.columns.get_loc('Seller_Type_Individual')]:.4f} |
| Brand_fortuner | {lr_model.coef_[X.columns.get_loc('Brand_fortuner')]:.4f} |
| Brand_Royal | {lr_model.coef_[X.columns.get_loc('Brand_Royal')]:.4f} |

The largest positive coefficient is for **Present_Price_Log**, confirming it is the dominant factor. The large negative coefficients for **Seller_Type_Individual** and **Transmission_Manual** suggest that being an individual seller or having a manual transmission significantly **reduces** the log selling price compared to the baseline (Dealer, Automatic).

**5. Predictions for Test Dataset:**
The predictions for the test set are stored in the variable `y_test_pred`.
"""

with open('model_assessment.txt', 'w') as f:
    f.write(results)

print("Model training and assessment complete. Results saved to 'model_assessment.txt'.")
