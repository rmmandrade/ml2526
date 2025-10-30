import pandas as pd

# Define the target and ID columns based on the new dataset structure
TARGET_COL = 'price'
ID_COL = 'carID'

# Load the datasets
try:
    train_df = pd.read_csv('/home/ubuntu/upload/train.csv')
    test_df = pd.read_csv('/home/ubuntu/upload/test.csv')
except FileNotFoundError:
    print("Error: train.csv or test.csv not found in the upload directory.")
    exit()

# Add a source column to identify the origin of the data
train_df['Source'] = 'train'
test_df['Source'] = 'test'

# Store test IDs for submission file
test_ids = test_df[ID_COL]
test_df['ID_for_submission'] = test_ids

# Check if the target column is in the test set (it should not be)
if TARGET_COL in test_df.columns:
    print(f"\nWarning: '{TARGET_COL}' found in test.csv. Dropping it.")
    test_df.drop(TARGET_COL, axis=1, inplace=True)

# Align columns before merging, keeping only common columns and the target in train
# The list of columns to keep in the train set is all common columns + the target column
common_cols = list(set(train_df.columns) & set(test_df.columns))
train_cols_to_keep = common_cols + [TARGET_COL]
train_df = train_df[train_cols_to_keep]

# Concatenate the datasets for consistent feature engineering and preprocessing
combined_df = pd.concat([train_df, test_df], ignore_index=True)

print("\nCombined DataFrame Info:")
print(combined_df.info())
print("\nCombined DataFrame Head:")
print(combined_df.head())

# Save the combined DataFrame and test IDs
combined_df.to_csv('combined_data.csv', index=False)
test_ids.to_csv('test_ids.csv', index=False)

print("\nCombined data saved to 'combined_data.csv'.")
print(f"Test IDs saved to 'test_ids.csv'. Total test samples: {len(test_ids)}")
