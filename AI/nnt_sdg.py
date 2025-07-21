import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

ages = np.random.normal(loc=30, scale=10, size=n)
salaries = np.random.normal(loc=50000, scale=10000, size=n)
salaries += (ages - 30) * 1000
departments = np.random.choice(['HR', 'IT', 'Sales', 'Marketing'], size=n)

df = pd.DataFrame({'ages': ages, 'salaries': salaries, 'departments': departments})

df.to_csv('salaries.csv', index=False)
print(df.head())

dates = pd.date_range(start='2024-01-01', periods=365, freq='D')
temperatures = 10 + 15 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 2, 365)

df_temp = pd.DataFrame({
    'Date': dates,
    'Temperature_C': temperatures
})

print(df_temp.head())

# 