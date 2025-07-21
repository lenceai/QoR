from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sdv.tabular import CTGAN

# Create original imbalanced dataset (similar to SMOTE example)
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.95, 0.05], random_state=42)

# Convert to DataFrame for CTGAN
df_original = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df_original['target'] = y

print("Before CTGAN:", df_original['target'].value_counts())

# Initialize and fit CTGAN
ctgan = CTGAN(
    primary_key=None,
    epochs=100,
    verbose=False,
    random_state=42
)

ctgan.fit(df_original)

# Generate synthetic data
df_synthetic = ctgan.sample(num_rows=len(df_original))

print("After CTGAN:", df_synthetic['target'].value_counts())

# Compare original vs synthetic data
print(f"\nOriginal dataset shape: {df_original.shape}")
print(f"Synthetic dataset shape: {df_synthetic.shape}")
print(f"Original target distribution:\n{df_original['target'].value_counts(normalize=True)}")
print(f"Synthetic target distribution:\n{df_synthetic['target'].value_counts(normalize=True)}") 