import pandas as pd
import numpy as np
import pickle
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Parameters
start_date = '1990-01-01'
end_date = '2024-01-01'
assets = ['SPX_Syn', 'FTSE_Syn', 'DAX_Syn']
seed = 42

np.random.seed(seed)

# Generate Date Range
dates = pd.date_range(start=start_date, end=end_date, freq='B') # Business days
n = len(dates)

# Generate Synthetic Returns (Student-t with 5 df to have fat tails, scaled)
# We want typical daily volatility around 1%
df_data = {}
for asset in assets:
    # t-dist returns
    returns = np.random.standard_t(df=5, size=n) * 0.01
    # Convert to prices (starting at 1000)
    prices = 1000 * np.cumprod(1 + returns)
    df_data[asset] = prices

df = pd.DataFrame(df_data, index=dates)

# Save to pickle as expected by experiments_indexes.py
output_path = 'data/indexes.pickle'
with open(output_path, 'wb') as f:
    pickle.dump(df, f)

print(f"Synthetic data generated and saved to {output_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date Range: {df.index.min()} to {df.index.max()}")
