import numpy as np
import pickle
import warnings
from models.gas import GAS1

# Suppress warnings
warnings.simplefilter(action='ignore')

# Load data
DATA_PATH = '../data/indexes.pickle'
with open(DATA_PATH, 'rb') as f:
    df = pickle.load(f)

# Compute log returns (decimal)
df_returns = np.log(df / df.shift(1)).dropna()
y = df_returns.iloc[:, 0].values  # Take first asset (SP500)
y = y[:2000]  # Train window

print(f"Data stats: Mean={np.mean(y):.5f}, Std={np.std(y):.5f}, Min={np.min(y):.5f}")

# Initialize GAS1
theta = 0.025
model = GAS1(theta)

# Manual Initialization Logic from GAS1.fit
n_emp = int(np.ceil(0.1 * len(y)))
y_sort = np.sort(y[:n_emp])
quantile0 = int(round(n_emp * theta))-1
if quantile0 < 0: quantile0 = 0
k0 = np.log(-y_sort[quantile0]) if y_sort[quantile0]<0 else np.log(y_sort[quantile0])

print(f"Initial k0: {k0:.5f} (implies exp(k0) = {np.exp(k0):.5f})")
print(f"Hardcoded beta0: {model.beta0 if hasattr(model, 'beta0') else '[-1.164, -1.757, 0.995, 0.007]'}")

# Check initial VaR with hardcoded beta0
beta0 = np.array([-1.164, -1.757, 0.995, 0.007])
q_init = beta0[0] * np.exp(k0)
print(f"Initial VaR (q0) with hardcoded beta: {q_init:.5f}")

# Fit model
print("\nFitting GAS1...")
res = model.fit(y, seed=42, return_train=True)
beta_opt = res['beta']
print(f"Optimized beta: {beta_opt}")

# Check In-Sample Performance
qi = res['qi']
violations = np.sum(y < qi)
rate = violations / len(y)
print(f"In-sample violation rate: {rate*100:.2f}% (Target: {theta*100}%)")
print(f"Mean VaR: {np.mean(qi):.5f}")
