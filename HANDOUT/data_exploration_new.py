import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the combined dataset
df = pd.read_csv('combined_data.csv')

# --- Task 1: Import the dataset and explore the data (3 points) ---

# 1. Check data contents, provide descriptive statistics and check for inconsistencies in the data.
print("--- Initial Data Exploration (Task 1) ---")
print("\nShape of the dataset (rows, columns):", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nData types and non-null counts:")
df.info()
print("\nDescriptive statistics for numerical columns:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values count per column:")
print(df.isnull().sum())

# Check for inconsistencies (e.g., non-positive values for price/mileage)
print("\nChecking for inconsistencies:")
# Price should be positive (only check non-NaN values)
print(f"Number of rows with non-positive price: {len(df[df['price'] <= 0])}")
# mileage should be non-negative
print(f"Number of rows with negative mileage: {len(df[df['mileage'] < 0])}")
# year should be reasonable
print(f"Min/Max Year: {df['year'].min()}/{df['year'].max()}")

# 2. Explore data visually and extract relevant insights.

# Create 'Car_Age' feature for visualization
df['Car_Age'] = 2025 - df['year']

# Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Distribution of the target variable (price)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'].dropna(), kde=True, bins=50)
plt.title('Distribution of Car Price')
plt.xlabel('Price')
plt.savefig('new_price_distribution.png')
plt.close()

# Price vs. Car Age
plt.figure(figsize=(10, 6))
sns.regplot(x='Car_Age', y='price', data=df, scatter_kws={'alpha':0.3})
plt.title('Price vs. Car Age')
plt.xlabel('Car Age (Years)')
plt.ylabel('Price')
plt.savefig('new_price_vs_age.png')
plt.close()

# Categorical features vs. Price
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.boxplot(x='transmission', y='price', data=df, ax=axes[0])
axes[0].set_title('Price by Transmission Type')
sns.boxplot(x='fuelType', y='price', data=df, ax=axes[1])
axes[1].set_title('Price by Fuel Type')
sns.boxplot(x='hasDamage', y='price', data=df, ax=axes[2])
axes[2].set_title('Price by Damage Status')
plt.tight_layout()
plt.savefig('new_categorical_vs_price.png')
plt.close()

# Save initial data info to a file for later report writing
with open('new_initial_data_summary.txt', 'w') as f:
    f.write("Initial Data Exploration Summary (New Dataset)\n")
    f.write(f"Shape: {df.shape}\n\n")
    f.write("Data Types and Missing Values:\n")
    df.info(buf=f)
    f.write("\nDescriptive Statistics:\n")
    f.write(df.describe(include='all').to_string())
    f.write("\n\nInconsistencies Check:\n")
    f.write(f"Non-positive price: {len(df[df['price'] <= 0])}\n")
    f.write(f"Negative mileage: {len(df[df['mileage'] < 0])}\n")
    f.write(f"Min/Max Year: {df['year'].min()}/{df['year'].max()}\n")

print("\nInitial data summary saved to 'new_initial_data_summary.txt'.")
print("Visualizations generated: new_price_distribution.png, new_price_vs_age.png, new_categorical_vs_price.png")
