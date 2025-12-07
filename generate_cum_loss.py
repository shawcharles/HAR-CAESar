import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from har_caesar.utils import patton_loss

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

# Load data
path = 'output/har_experiment/results_theta001.pickle'
if not os.path.exists(path):
    print("Results file not found.")
    exit()

with open(path, 'rb') as f:
    res = pickle.load(f)

theta = 0.01
models = res['models']
assets = ['SP500', 'FTSE100', 'Nikkei225', 'MSCI_EM']

# We need to stitch together the full time series for all assets
# to compute a representative cumulative loss.
# Alternatively, we can just pick one representative asset (e.g. SP500)
# or average across assets. Let's do SP500 as it's the primary benchmark.

asset = 'SP500'
windows = res['windows']

# Collect all forecasts and true values for SP500
y_all = []
q_caesar_all = []
e_caesar_all = []
e_har_all = []
dates_all = []

# Sort windows just in case
n_windows = len(windows)
# Note: windows overlap? No, experiments_har_caesar.py uses rolling windows 
# where test set is separate. 
# Window 0: Test [2000:2250]
# Window 1: Test [2250:2500]
# So they are sequential and non-overlapping in the test set.

for window_idx in range(n_windows):
    # Find data for this window/asset
    y_w = None
    for a, w, y in models['CAESar']['y_true']:
        if a == asset and w == window_idx:
            y_w = y
            break
            
    if y_w is None: continue
    
    # Get test start date
    start_date = windows[window_idx]['test_start']
    dates_w = pd.date_range(start=start_date, periods=len(y_w), freq='B')
    
    # Get forecasts
    q_c = None
    e_c = None
    e_h = None
    
    for a, w, q in models['CAESar']['qf']:
        if a == asset and w == window_idx: q_c = q; break
    for a, w, e in models['CAESar']['ef']:
        if a == asset and w == window_idx: e_c = e; break
    for a, w, e in models['HAR_CAESar']['ef']:
        if a == asset and w == window_idx: e_h = e; break
        
    if q_c is not None and e_c is not None and e_h is not None:
        y_all.extend(y_w)
        q_caesar_all.extend(q_c.flatten())
        e_caesar_all.extend(e_c.flatten())
        e_har_all.extend(e_h.flatten())
        dates_all.extend(dates_w)

# Convert to arrays
y_all = np.array(y_all)
q_caesar_all = np.array(q_caesar_all)
e_caesar_all = np.array(e_caesar_all)
e_har_all = np.array(e_har_all)
dates_all = np.array(dates_all)

# Compute Loss Per Day
# We use the FZ loss function
def fz_loss_elementwise(y, q, e, theta):
    return np.where(y <= q, (y - q) / (theta * e), 0) + q / e + np.log(-e) - 1

loss_caesar = fz_loss_elementwise(y_all, q_caesar_all, e_caesar_all, theta)
loss_har = fz_loss_elementwise(y_all, q_caesar_all, e_har_all, theta) # Uses same Q (Caesar Q = HAR Q)

# Compute Differential: (Loss HAR) - (Loss CAESar)
# Positive value => HAR is WORSE (higher loss)
# Negative value => HAR is BETTER (lower loss)
diff = loss_har - loss_caesar
cum_diff = np.cumsum(diff)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(dates_all, cum_diff, color='darkblue', linewidth=1.5)

# Add reference line at 0
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Highlight regimes
# When line goes UP: HAR is underperforming (costing premium)
# When line goes DOWN: HAR is outperforming (paying out)

ax.set_title('Cumulative Loss Differential (HAR-CAESar minus CAESar)', fontsize=16)
ax.set_ylabel('Cumulative FZ Loss Difference')
ax.set_xlabel('Year')

# Annotate GFC and COVID
# Find indices for 2008 and 2020
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator(2))

# Shade the "Premium" zones (calm periods where line drifts up) vs "Payout" zones (crises)
# This might be too cluttered, let's just stick to the line.
# It clearly shows the narrative.

plt.tight_layout()
os.makedirs('output/plots', exist_ok=True)
plt.savefig('output/plots/cumulative_loss_diff.png', dpi=300)
plt.savefig('output/plots/cumulative_loss_diff.pdf')
print("Plot saved to output/plots/cumulative_loss_diff.png")
