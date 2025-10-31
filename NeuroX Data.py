# Code to Generate Synthetic Datasets for NeuroDetect-X Project
# This script generates two synthetic datasets:
# 1. Primary Dataset: Mimics the main dataset with 2847 samples, 156 features, and class distribution as per the paper.
# 2. Validation Dataset: Mimics the UCI Parkinson's Speech Dataset with 198 samples, 22 features, binary classes (PD vs. Healthy).
# Features are generated with some realism: motor, clinical, biomarker features with added noise and correlations.
# Classes: 0 = Normal Controls, 1 = Parkinson's Disease, 2 = Essential Tremor, 3 = Other NMDs (for primary); 0 = Healthy, 1 = PD (for validation).
# Save as CSV files for use in the project.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)  # For reproducibility

# Function to generate features with categories
def generate_features(n_samples, n_features, categories, noise_level=0.1):
    features = []
    for cat, count in categories.items():
        # Generate base features with some correlation within categories
        base = np.random.randn(count, n_features)
        # Add category-specific shifts
        shift = np.random.randn(n_features) * 0.5
        base += shift
        # Add noise
        noise = np.random.randn(count, n_features) * noise_level
        base += noise
        features.append(base)
    return np.vstack(features)

# Primary Dataset Generation
n_samples_primary = 2847
n_features_primary = 156
class_distribution = {
    0: 1247,  # Normal Controls (43.8%)
    1: 892,   # Parkinson's Disease (31.3%)
    2: 398,   # Essential Tremor (14.0%)
    3: 310    # Other NMDs (10.9%)
}

# Feature categories (simplified): Motor (50%), Clinical (30%), Biomarkers (20%)
motor_features = 78
clinical_features = 47
biomarker_features = 31

# Generate features
X_primary = generate_features(n_samples_primary, n_features_primary, class_distribution)

# Assign classes
y_primary = []
for cls, count in class_distribution.items():
    y_primary.extend([cls] * count)
y_primary = np.array(y_primary)

# Shuffle to mix classes
indices = np.random.permutation(n_samples_primary)
X_primary = X_primary[indices]
y_primary = y_primary[indices]

# Scale features
scaler = StandardScaler()
X_primary_scaled = scaler.fit_transform(X_primary)

# Create DataFrame
columns = [f'feature_{i}' for i in range(n_features_primary)]
df_primary = pd.DataFrame(X_primary_scaled, columns=columns)
df_primary['target'] = y_primary

# Save to CSV
df_primary.to_csv('primary_dataset.csv', index=False)
print("Primary dataset saved as 'primary_dataset.csv' with shape:", df_primary.shape)

# Validation Dataset Generation (UCI-like: 198 samples, 22 features, binary)
n_samples_validation = 198
n_features_validation = 22
class_distribution_val = {
    0: 8,   # Healthy Controls
    1: 23   # Parkinson's Disease
}

# Features: Biomedical voice features (e.g., Jitter, Shimmer, DFA, PPE)
X_validation = generate_features(n_samples_validation, n_features_validation, class_distribution_val)

# Assign classes
y_validation = []
for cls, count in class_distribution_val.items():
    y_validation.extend([cls] * count)
y_validation = np.array(y_validation)

# Shuffle
indices_val = np.random.permutation(n_samples_validation)
X_validation = X_validation[indices_val]
y_validation = y_validation[indices_val]

# Scale
X_validation_scaled = scaler.fit_transform(X_validation)

# Create DataFrame
columns_val = [f'voice_feature_{i}' for i in range(n_features_validation)]
df_validation = pd.DataFrame(X_validation_scaled, columns=columns_val)
df_validation['target'] = y_validation

# Save to CSV
df_validation.to_csv('validation_dataset.csv', index=False)
print("Validation dataset saved as 'validation_dataset.csv' with shape:", df_validation.shape)

# Optional: Print class distributions
print("\nPrimary Dataset Class Distribution:")
print(df_primary['target'].value_counts(normalize=True))

print("\nValidation Dataset Class Distribution:")
print(df_validation['target'].value_counts(normalize=True))
