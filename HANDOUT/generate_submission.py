import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Load the processed training and testing datasets
df_train = pd.read_csv('processed_train.csv')
df_test = pd.read_csv('processed_test.csv')

# Load the final feature list
with open('new_final_features.txt', 'r') as f:
    final_features = [line.strip() for line in f.readlines()]

# Load the test IDs from the original test file to ensure correct length and index
test_df_original = pd.read_csv('/home/ubuntu/upload/test.csv')
test_ids = test_df_original['carID']

# --- Train the Final Model ---
X_train = df_train[final_features]
y_train = df_train['price_log']

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# --- Prepare Test Data for Prediction ---
# Ensure the test data has all the required features, padding with 0s if a feature was only in the training set
missing_cols = set(final_features) - set(df_test.columns)
for c in missing_cols:
    df_test[c] = 0

# Select and align the features in the test set
X_test = df_test[final_features]

# --- Generate Predictions ---
y_test_pred_log = lr_model.predict(X_test)

# Inverse transform the log-transformed predictions to get the final price
y_test_pred = np.expm1(y_test_pred_log)

# --- Create Submission File ---
submission_df = pd.DataFrame({
    'carID': test_ids.values, # Use .values to ensure array length matches prediction length
    'price': y_test_pred
})

# Round the price to 2 decimal places
submission_df['price'] = submission_df['price'].round(2)

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
print("First 5 predictions:")
print(submission_df.head())
