from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import pandas as pd

X, y = make_classification(n_samples=1000, n_features=10, weights=[0.95, 0.05], random_state=42)
print("Before SMOTE:", pd.Series(y).value_counts())

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("After SMOTE:", pd.Series(y_res).value_counts())