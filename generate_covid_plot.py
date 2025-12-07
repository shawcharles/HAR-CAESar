import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

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

# Find SP500 COVID window
target_date = pd.Timestamp('2020-03-16')
windows = res['windows']
target_window_idx = -1
for i, w in enumerate(windows):
    if w['test_start'] <= target_date <= w['test_end']:
        target_window_idx = i
        break

if target_window_idx == -1:
    print("Target window not found.")
    exit()

# Extract data
asset = 'SP500'
models = res['models']

# Get true returns
y_true = None
for a, w, y in models['CAESar']['y_true']:
    if a == asset and w == target_window_idx:
        y_true = y
        break

# Get forecasts
q_caesar = None
e_caesar = None
e_har = None

for a, w, q in models['CAESar']['qf']:
    if a == asset and w == target_window_idx:
        q_caesar = q
        break
for a, w, e in models['CAESar']['ef']:
    if a == asset and w == target_window_idx:
        e_caesar = e
        break
for a, w, e in models['HAR_CAESar']['ef']:
    if a == asset and w == target_window_idx:
        e_har = e
        break

# Create Date Range
start_date = windows[target_window_idx]['test_start']
dates = pd.date_range(start=start_date, periods=len(y_true), freq='B')

# Filter for plotting (Feb 2020 - April 2020)
plot_start = pd.Timestamp('2020-02-01')
plot_end = pd.Timestamp('2020-04-15')
mask = (dates >= plot_start) & (dates <= plot_end)

dates_plot = dates[mask]
y_plot = y_true[mask]
q_caesar_plot = q_caesar[mask]
e_caesar_plot = e_caesar[mask]
e_har_plot = e_har[mask]

# Create Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Returns
ax.bar(dates_plot, y_plot, color='gray', alpha=0.3, label='S&P 500 Returns', width=1.0)

# Plot Forecasts
ax.plot(dates_plot, q_caesar_plot, color='black', linestyle='--', linewidth=1.5, label='VaR (Both Models)')
ax.plot(dates_plot, e_caesar_plot, color='blue', linestyle='-', linewidth=2, label='CAESar ES')
ax.plot(dates_plot, e_har_plot, color='red', linestyle='-', linewidth=2, label='HAR-CAESar ES')

# Highlight the Crash Day
crash_date = pd.Timestamp('2020-03-16')
crash_idx = np.where(dates_plot == crash_date)[0]
if len(crash_idx) > 0:
    ax.axvline(crash_date, color='black', alpha=0.2, linestyle=':')
    # Add text annotation
    # ax.annotate('March 16 Crash', xy=(crash_date, -0.13), xytext=(crash_date, -0.20),
    #             arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

# Formatting
ax.set_title('Forecast Reaction to COVID-19 Crash (S&P 500)', fontsize=16, pad=15)
ax.set_ylabel('Log Returns / Forecasts')
ax.legend(loc='lower left', frameon=True)
ax.set_ylim(-0.25, 0.10)

# Date formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=0)

plt.tight_layout()
os.makedirs('output/plots', exist_ok=True)
plt.savefig('output/plots/covid_crash.png', dpi=300)
plt.savefig('output/plots/covid_crash.pdf')
print("Plot saved to output/plots/covid_crash.png")
