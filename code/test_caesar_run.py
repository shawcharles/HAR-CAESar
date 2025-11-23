import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from models.caesar import CAESar
import os

# Load data
print("Loading data...")
with open('../data/indexes.pickle', 'rb') as f:
    df = pickle.load(f)

# Compute returns
returns = df.pct_change().dropna()
y = returns['SPX_Syn'].values # Select one asset

# Parameters
theta = 0.025
split_point = 2000 # Train on first 2000 days, predict next
test_size = 500 # Predict next 500 days
y_subset = y[:split_point + test_size]

print(f"Running CAESar on synthetic SPX data...")
print(f"Train size: {split_point}")
print(f"Test size: {test_size}")
print(f"Theta: {theta}")

# Initialize and run model
# 'AS' is the Asymmetric Slope specification (standard)
model = CAESar(theta=theta, spec='AS') 

print("Fitting model (this may take a moment)...")
# fit_predict takes the whole series and the split point 'tv'
results = model.fit_predict(y_subset, ti=split_point, seed=42)

print("Model run complete.")

# Extract forecasts
q_pred = results['qf'] # VaR forecasts (out of sample)
e_pred = results['ef'] # ES forecasts (out of sample)
y_test = y_subset[split_point:]

# Basic Evaluation
violations = np.sum(y_test < q_pred)
violation_rate = violations / len(y_test)

print("\n--- Results ---")
print(f"Violation Rate: {violation_rate:.4f} (Target: {theta})")
print(f"Mean VaR: {np.mean(q_pred):.4f}")
print(f"Mean ES: {np.mean(e_pred):.4f}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Returns', color='grey', alpha=0.5)
plt.plot(q_pred, label=f'VaR {theta}', color='blue')
plt.plot(e_pred, label=f'ES {theta}', color='red')
plt.title(f'CAESar Forecasts (Synthetic Data, theta={theta})')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot
os.makedirs('../output', exist_ok=True)
plot_path = '../output/test_caesar_plot.png'
plt.savefig(plot_path)
print(f"\nPlot saved to {plot_path}")
