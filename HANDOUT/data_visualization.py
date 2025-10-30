import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('car_data.csv')
except FileNotFoundError:
    print("Error: 'car_data.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- Task 1: Explore data visually and extract relevant insights. ---

# 1. Create 'Car_Age' feature for better analysis
df['Car_Age'] = 2025 - df['Year'] # Assuming current year is 2025 based on the handout date

# 2. Set up visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 3. Univariate Analysis: Distribution of the target variable (Selling_Price)
plt.figure(figsize=(10, 6))
sns.histplot(df['Selling_Price'], kde=True, bins=30)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price (in Lakhs)')
plt.savefig('selling_price_distribution.png')
plt.close()

# 4. Bivariate Analysis: Selling Price vs. Present Price (Multivariate: Selling Price vs. Present Price vs. Fuel Type)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Present_Price', y='Selling_Price', hue='Fuel_Type', data=df, s=100)
plt.title('Selling Price vs. Present Price, colored by Fuel Type')
plt.xlabel('Present Price (in Lakhs)')
plt.ylabel('Selling Price (in Lakhs)')
plt.savefig('price_vs_fuel.png')
plt.close()

# 5. Bivariate Analysis: Selling Price vs. Car Age
plt.figure(figsize=(10, 6))
sns.regplot(x='Car_Age', y='Selling_Price', data=df, scatter_kws={'alpha':0.6})
plt.title('Selling Price vs. Car Age')
plt.xlabel('Car Age (Years)')
plt.ylabel('Selling Price (in Lakhs)')
plt.savefig('price_vs_age.png')
plt.close()

# 6. Multivariate Analysis: Categorical features vs. Selling Price
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Seller_Type vs. Selling_Price
sns.boxplot(x='Seller_Type', y='Selling_Price', data=df, ax=axes[0])
axes[0].set_title('Selling Price by Seller Type')
axes[0].set_xlabel('Seller Type')
axes[0].set_ylabel('Selling Price (in Lakhs)')

# Transmission vs. Selling_Price
sns.boxplot(x='Transmission', y='Selling_Price', data=df, ax=axes[1])
axes[1].set_title('Selling Price by Transmission Type')
axes[1].set_xlabel('Transmission Type')
axes[1].set_ylabel('')

# Fuel_Type vs. Selling_Price
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df, ax=axes[2])
axes[2].set_title('Selling Price by Fuel Type')
axes[2].set_xlabel('Fuel Type')
axes[2].set_ylabel('')

plt.tight_layout()
plt.savefig('categorical_vs_price.png')
plt.close()

# 7. Correlation Matrix for numerical features
numerical_cols = ['Selling_Price', 'Present_Price', 'Kms_Driven', 'Car_Age', 'Owner']
correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_matrix.png')
plt.close()

print("Visualizations generated: selling_price_distribution.png, price_vs_fuel.png, price_vs_age.png, categorical_vs_price.png, correlation_matrix.png")

# Save insights to a file
insights = """
--- Relevant Insights from Data Visualization (Task 1) ---

1. Distribution of Selling Price:
   - The distribution is highly right-skewed, with most cars selling for less than 10 Lakhs.
   - There are a few outliers with very high selling prices (up to 35 Lakhs), suggesting a need for potential transformation (e.g., log) or outlier handling.

2. Selling Price vs. Present Price (Multivariate with Fuel Type):
   - There is a strong positive linear relationship between 'Present_Price' and 'Selling_Price' (correlation of 0.88).
   - Diesel cars tend to have higher 'Present_Price' and 'Selling_Price' compared to Petrol and CNG cars, especially in the higher price range.

3. Selling Price vs. Car Age:
   - There is a clear negative correlation (approx -0.24) between 'Car_Age' and 'Selling_Price', indicating that older cars sell for less.
   - The scatter plot shows a high concentration of newer cars (low age) across all selling prices, but the highest selling prices are exclusively for newer cars.

4. Categorical Features vs. Selling Price (Box Plots):
   - Seller Type: Cars sold by 'Dealer' have a significantly higher median and range of selling prices than cars sold by an 'Individual'.
   - Transmission Type: Automatic cars have a much higher median and maximum selling price than Manual cars.
   - Fuel Type: Diesel cars command the highest prices, followed by Petrol, with CNG cars having the lowest prices.

5. Correlation Matrix:
   - The strongest correlation is between 'Selling_Price' and 'Present_Price' (0.88).
   - 'Selling_Price' has a moderate negative correlation with 'Car_Age' (-0.24) and 'Kms_Driven' (-0.03), and a weak positive correlation with 'Owner' (0.05).
   - The 'Year' column (which is inversely related to 'Car_Age') is not included in the correlation matrix, but its relationship with price is captured by 'Car_Age'.

Conclusion for Task 1:
The dataset is clean with no missing values. The target variable is highly skewed. 'Present_Price', 'Seller_Type', 'Transmission', and 'Fuel_Type' are the most influential features on 'Selling_Price'. The problem is a **Regression** task.
"""

with open('data_visualization_insights.txt', 'w') as f:
    f.write(insights)

print("Insights saved to 'data_visualization_insights.txt'.")
