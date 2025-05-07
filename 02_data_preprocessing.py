# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
print("Loading dataset...")
column_names = ['label'] + [f'feature_{i}' for i in range(1, 29)]
data = pd.read_csv('HIGGS.csv.gz', names=column_names, compression='gzip')

# Split features and target variable
X = data.drop('label', axis=1)
y = data['label']

# Check class distribution before handling imbalance
print("\nClass distribution BEFORE balancing:")
print(y.value_counts())

# Visualize initial class distribution
sns.countplot(x=y, palette='viridis')
plt.title('Initial Class Distribution')
plt.xlabel('Class (0 = Background, 1 = Signal)')
plt.ylabel('Count')
plt.savefig('initial_class_distribution.png')
plt.show()

# Handle class imbalance (random undersampling)
print("\nHandling class imbalance with Random Undersampling...")
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Verify new class distribution
print("\nClass distribution AFTER balancing:")
print(pd.Series(y_resampled).value_counts())

# Visualize balanced class distribution
sns.countplot(x=y_resampled, palette='viridis')
plt.title('Balanced Class Distribution (After Undersampling)')
plt.xlabel('Class (0 = Background, 1 = Signal)')
plt.ylabel('Count')
plt.savefig('balanced_class_distribution.png')
plt.show()

# Split the balanced data into training and test sets (80% train, 20% test)
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Feature scaling (standardization)
print("\nStandardizing features...")
scaler = StandardScaler()

# Fit scaler on training data only
X_train_scaled = scaler.fit_transform(X_train)

# Apply scaler to test data
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data to numpy files for easy loading later
print("\nSaving processed data to disk...")
np.save('X_train.npy', X_train_scaled)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test_scaled)
np.save('y_test.npy', y_test)

print("\nData preprocessing completed successfully.")
