import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('car_data.csv')
except FileNotFoundError:
    print("Error: 'car_data.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- Task 1: Import the dataset and explore the data (3 points) ---

# 1. Check data contents, provide descriptive statistics and check for inconsistencies in the data.
print("--- Initial Data Exploration (Task 1) ---")
print("\nShape of the dataset (rows, columns):", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nData types and non-null counts:")
df.info()
print("\nDescriptive statistics for numerical columns:")
print(df.describe())

# Check for inconsistencies (e.g., non-positive values for price/kms_driven, min/max values)
print("\nChecking for inconsistencies:")
# Selling_Price and Present_Price should be positive
print(f"Number of rows with non-positive Selling_Price: {len(df[df['Selling_Price'] <= 0])}")
print(f"Number of rows with non-positive Present_Price: {len(df[df['Present_Price'] <= 0])}")
# Kms_Driven should be non-negative
print(f"Number of rows with negative Kms_Driven: {len(df[df['Kms_Driven'] < 0])}")

# 2. Explore data visually and extract relevant insights. Explain your rationale and findings. Do not forget to analyse multivariate relationships.
# This part will be done in a separate step with matplotlib/seaborn to generate visualizations.

# Save initial data info to a file for later report writing
with open('initial_data_summary.txt', 'w') as f:
    f.write("Initial Data Exploration Summary\n")
    f.write(f"Shape: {df.shape}\n\n")
    f.write("Data Types:\n")
    df.info(buf=f)
    f.write("\nDescriptive Statistics:\n")
    f.write(df.describe().to_string())
    f.write("\n\nInconsistencies Check:\n")
    f.write(f"Non-positive Selling_Price: {len(df[df['Selling_Price'] <= 0])}\n")
    f.write(f"Non-positive Present_Price: {len(df[df['Present_Price'] <= 0])}\n")
    f.write(f"Negative Kms_Driven: {len(df[df['Kms_Driven'] < 0])}\n")

print("\nInitial data summary saved to 'initial_data_summary.txt'.")

# Next step will be visualization and multivariate analysis.
# For now, we will execute this script to get the initial text output.
