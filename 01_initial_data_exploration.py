# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Load the dataset
print("Loading dataset...")
column_names = ['label'] + [f'feature_{i}' for i in range(1, 29)]
data = pd.read_csv('HIGGS.csv.gz', names=column_names, compression='gzip')

# Display basic information about the dataset
print("\nDataset Information:")
print(data.info())

# Display first few rows
print("\nFirst 5 rows:")
print(data.head())

# Step 2: Check for missing values
print("\nChecking for missing values:")
print(data.isnull().sum())

# Step 3: Class Distribution
print("\nClass distribution (0 = Background, 1 = Signal):")
print(data['label'].value_counts())

# Visualize class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='label', data=data, palette='viridis')
plt.title('Class Distribution: Higgs Signal vs Background')
plt.xlabel('Class (0 = Background, 1 = Signal)')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.show()

# Step 4: Statistical summary of features
print("\nStatistical summary of features:")
print(data.describe())

# Step 5: Correlation matrix (heatmap)
print("\nGenerating correlation matrix...")
correlation_matrix = data.sample(10000, random_state=42).corr()  # sampling to reduce memory usage
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap (Sampled 10,000 records)')
plt.savefig('correlation_matrix.png')
plt.show()
