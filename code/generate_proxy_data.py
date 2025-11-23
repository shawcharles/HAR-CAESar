import pandas as pd
import numpy as np
import pickle
import os

# Ensure data directory exists
os.makedirs('../data', exist_ok=True)

# Parameters
start_date = '2000-01-01'
end_date = '2023-12-31'
seed = 42
np.random.seed(seed)

# Generate Date Range (Business Days)
dates = pd.date_range(start=start_date, end=end_date, freq='B')
n = len(dates)

def generate_series(n, df_t, vol, drift=0.0002):
    """
    Generate a price series with Student-t returns (heavy tails).
    df_t: degrees of freedom (lower = heavier tails)
    vol: daily volatility (standard deviation)
    drift: daily drift (expected return)
    """
    # Standardised t-distributed innovations
    innovations = np.random.standard_t(df=df_t, size=n)
    # Scale to desired volatility. std of t(df) is sqrt(df/(df-2)) for df>2
    t_std = np.sqrt(df_t / (df_t - 2))
    scaled_returns = (innovations / t_std) * vol + drift
    
    # Add some volatility clustering (GARCH-like effect)
    # Simple stochastic volatility proxy: random walk in log-vol
    log_vol = np.zeros(n)
    log_vol[0] = np.log(vol)
    vol_vol = 0.05 # volatility of volatility
    for t in range(1, n):
        log_vol[t] = 0.98 * log_vol[t-1] + 0.02 * np.log(vol) + np.random.normal(0, vol_vol)
    
    dynamic_vol = np.exp(log_vol)
    final_returns = (innovations / t_std) * dynamic_vol + drift
    
    # Convert to prices
    prices = 100.0 * np.cumprod(1 + final_returns)
    return prices

# 1. S&P 500 Proxy (Main)
# Moderate tails (df=5), Vol ~1.0-1.2% daily
spx = generate_series(n, df_t=6, vol=0.011)

# 2. FTSE 100 Proxy (Robustness 1)
# Similar to SPX but slightly different drift/vol
ftse = generate_series(n, df_t=6, vol=0.010, drift=0.00015)

# 3. Nikkei 225 Proxy (Robustness 2)
# Often slightly higher vol
n225 = generate_series(n, df_t=5, vol=0.013)

# 4. Emerging Markets Proxy (Robustness 3 - Stress)
# Heavy tails (df=3.5), High Vol (~1.5% daily)
em = generate_series(n, df_t=3.5, vol=0.016)

# Create DataFrame
data = {
    'SP500': spx,
    'FTSE100': ftse,
    'Nikkei225': n225,
    'MSCI_EM': em
}

df = pd.DataFrame(data, index=dates)

# Save to pickle
output_path = '../data/indexes.pickle' # Overwriting the main input file for convenience
with open(output_path, 'wb') as f:
    pickle.dump(df, f)

print(f"Proxy datasets generated and saved to {output_path}")
print(f"Series: {list(data.keys())}")
print(f"Observations: {n}")
print("\nDescriptive Stats of Returns:")
print(df.pct_change().describe())
