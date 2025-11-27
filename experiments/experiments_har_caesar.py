"""
HAR-CAESar Experiment Script

This script runs a comparative experiment between:
1. CAESar (baseline)
2. HAR-CAESar (proposed extension)
3. CAViaR (VaR-only baseline)
4. GAS models (GAS1, GAS2)

on four equity indices: S&P 500, FTSE 100, Nikkei 225, MSCI EM (EEM).

Author: MSc Thesis Implementation
"""

#%% Imports and Setup

import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
warnings.simplefilter(action='ignore')

# Import models
from har_caesar.models.caesar import CAESar
from har_caesar.models.har_caesar import HAR_CAESar
from har_caesar.models.caviar import CAViaR
from har_caesar.models.gas import GAS1, GAS2

# Configuration
OUTPUT_PATH = 'output/har_experiment'
DATA_PATH = 'data/indexes.pickle'
SEED = 42

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

#%% Load and Prepare Data

print("Loading data...")
with open(DATA_PATH, 'rb') as f:
    df = pickle.load(f)

# Rename columns for clarity
column_map = {
    '^GSPC': 'SP500',
    '^FTSE': 'FTSE100',
    '^N225': 'Nikkei225',
    'EEM': 'MSCI_EM'
}
df.columns = [column_map.get(c, c) for c in df.columns]

# Compute log returns
df_returns = np.log(df / df.shift(1)).dropna()

# Data summary
print(f"Data shape: {df_returns.shape}")
print(f"Date range: {df_returns.index[0]} to {df_returns.index[-1]}")
print(f"Columns: {list(df_returns.columns)}")

#%% Experiment Configuration

# Quantile levels
THETA_LEVELS = [0.025, 0.01]  # 2.5% and 1% VaR

# Models to run
MODELS = ['CAViaR', 'CAESar', 'HAR_CAESar', 'GAS1', 'GAS2']

# Rolling window parameters
TRAIN_WINDOW = 2000  # Training window size (~8 years)
TEST_WINDOW = 250    # Test window size (~1 year)
STEP_SIZE = 250      # Step between windows (~1 year)

#%% Helper Functions

def compute_violation_rate(y, q):
    """Compute empirical violation rate."""
    violations = np.sum(y < q)
    return violations / len(y)

def kupiec_test(y, q, theta):
    """Kupiec unconditional coverage test."""
    n = len(y)
    violations = np.sum(y < q)
    pi_hat = violations / n
    
    if violations == 0 or violations == n:
        return np.nan, np.nan
    
    # Log-likelihood ratio
    lr = -2 * (np.log(theta**violations * (1-theta)**(n-violations)) -
               np.log(pi_hat**violations * (1-pi_hat)**(n-violations)))
    
    # P-value (chi-squared with 1 df)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(lr, 1)
    
    return lr, p_value

def fissler_ziegel_loss(y, q, e, theta):
    """Compute mean Fissler-Ziegel loss."""
    # Ensure e is negative (ES convention)
    loss = np.where(y <= q, (y - q) / (theta * e), 0) + q / e + np.log(-e)
    return np.mean(loss)

def tick_loss(y, q, theta):
    """Compute mean tick (quantile) loss."""
    loss = (y - q) * (theta - (y < q).astype(float))
    return np.mean(loss)

#%% Main Experiment Loop

def run_experiment(theta):
    """Run experiment for a given theta level."""
    print(f"\n{'='*60}")
    print(f"Running experiment for theta = {theta}")
    print(f"{'='*60}")
    
    results = {
        'theta': theta,
        'models': {},
        'windows': []
    }
    
    # Initialize results structure for each model
    for model_name in MODELS:
        results['models'][model_name] = {
            'times': [],
            'qf': [],  # VaR forecasts
            'ef': [],  # ES forecasts (None for CAViaR)
            'y_true': []  # True returns
        }
    
    # Rolling window experiment
    n_windows = (len(df_returns) - TRAIN_WINDOW - TEST_WINDOW) // STEP_SIZE + 1
    
    # NOTE: The min(n_windows, 10) cap is for development/testing only.
    # For full experiments reported in thesis, remove the cap: range(n_windows)
    for window_idx in tqdm(range(min(n_windows, 10)), desc=f"Windows (theta={theta})"):
        start_idx = window_idx * STEP_SIZE
        train_end = start_idx + TRAIN_WINDOW
        test_end = train_end + TEST_WINDOW
        
        if test_end > len(df_returns):
            break
        
        window_info = {
            'train_start': df_returns.index[start_idx],
            'train_end': df_returns.index[train_end-1],
            'test_start': df_returns.index[train_end],
            'test_end': df_returns.index[min(test_end-1, len(df_returns)-1)]
        }
        results['windows'].append(window_info)
        
        # Run on each asset
        for asset in df_returns.columns:
            print(f"\n  Window {window_idx}, Asset: {asset}")
            
            # Get data
            y_full = df_returns[asset].values[start_idx:test_end]
            y_train = y_full[:TRAIN_WINDOW]
            y_test = y_full[TRAIN_WINDOW:]
            
            # Store true test values
            results['models']['CAESar']['y_true'].append((asset, window_idx, y_test))
            
            # --- CAViaR ---
            try:
                print(f"    Running CAViaR...")
                start_time = time.time()
                mdl = CAViaR(theta, 'AS', p=1, u=1)
                res = mdl.fit_predict(y_full, TRAIN_WINDOW, seed=SEED, return_train=False)
                elapsed = time.time() - start_time
                
                results['models']['CAViaR']['times'].append((asset, window_idx, elapsed))
                results['models']['CAViaR']['qf'].append((asset, window_idx, res['qf']))
                results['models']['CAViaR']['ef'].append((asset, window_idx, None))
                print(f"      Done in {elapsed:.1f}s")
            except Exception as e:
                print(f"      CAViaR failed: {e}")
            
            # --- CAESar ---
            try:
                print(f"    Running CAESar...")
                start_time = time.time()
                mdl = CAESar(theta, 'AS')
                res = mdl.fit_predict(y_full, TRAIN_WINDOW, seed=SEED, return_train=False)
                elapsed = time.time() - start_time
                
                results['models']['CAESar']['times'].append((asset, window_idx, elapsed))
                results['models']['CAESar']['qf'].append((asset, window_idx, res['qf']))
                results['models']['CAESar']['ef'].append((asset, window_idx, res['ef']))
                print(f"      Done in {elapsed:.1f}s")
            except Exception as e:
                print(f"      CAESar failed: {e}")
            
            # --- HAR-CAESar ---
            try:
                print(f"    Running HAR-CAESar...")
                start_time = time.time()
                mdl = HAR_CAESar(theta)
                res = mdl.fit_predict(y_full, TRAIN_WINDOW, seed=SEED, return_train=False)
                elapsed = time.time() - start_time
                
                results['models']['HAR_CAESar']['times'].append((asset, window_idx, elapsed))
                results['models']['HAR_CAESar']['qf'].append((asset, window_idx, res['qf']))
                results['models']['HAR_CAESar']['ef'].append((asset, window_idx, res['ef']))
                print(f"      Done in {elapsed:.1f}s")
            except Exception as e:
                print(f"      HAR-CAESar failed: {e}")
            
            # --- GAS1 ---
            try:
                print(f"    Running GAS1...")
                start_time = time.time()
                mdl = GAS1(theta)
                res = mdl.fit_predict(y_full, TRAIN_WINDOW, seed=SEED)
                elapsed = time.time() - start_time
                
                results['models']['GAS1']['times'].append((asset, window_idx, elapsed))
                results['models']['GAS1']['qf'].append((asset, window_idx, res['qf']))
                results['models']['GAS1']['ef'].append((asset, window_idx, res['ef']))
                print(f"      Done in {elapsed:.1f}s")
            except Exception as e:
                print(f"      GAS1 failed: {e}")
            
            # --- GAS2 ---
            try:
                print(f"    Running GAS2...")
                start_time = time.time()
                mdl = GAS2(theta)
                res = mdl.fit_predict(y_full, TRAIN_WINDOW, seed=SEED)
                elapsed = time.time() - start_time
                
                results['models']['GAS2']['times'].append((asset, window_idx, elapsed))
                results['models']['GAS2']['qf'].append((asset, window_idx, res['qf']))
                results['models']['GAS2']['ef'].append((asset, window_idx, res['ef']))
                print(f"      Done in {elapsed:.1f}s")
            except Exception as e:
                print(f"      GAS2 failed: {e}")
        
        # Save intermediate results
        save_path = f'{OUTPUT_PATH}/results_theta{str(theta).replace(".", "")}.pickle'
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n  Saved intermediate results to {save_path}")
    
    return results

#%% Compute Summary Statistics

def compute_summary(results):
    """Compute summary statistics from experiment results."""
    theta = results['theta']
    summary = {
        'theta': theta,
        'models': {}
    }
    
    for model_name in MODELS:
        model_data = results['models'][model_name]
        
        if len(model_data['qf']) == 0:
            continue
        
        # Aggregate results across assets and windows
        all_violations = []
        all_fz_loss = []
        all_tick_loss = []
        all_times = []
        
        for i, (asset, window_idx, qf) in enumerate(model_data['qf']):
            # Get corresponding true values
            y_true = None
            for a, w, y in results['models']['CAESar']['y_true']:
                if a == asset and w == window_idx:
                    y_true = y
                    break
            
            if y_true is None or len(qf) != len(y_true):
                continue
            
            # Violation rate
            vr = compute_violation_rate(y_true, qf)
            all_violations.append(vr)
            
            # Tick loss
            tl = tick_loss(y_true, qf, theta)
            all_tick_loss.append(tl)
            
            # FZ loss (if ES available)
            if model_data['ef'][i][2] is not None:
                ef = model_data['ef'][i][2]
                if len(ef) == len(y_true):
                    fz = fissler_ziegel_loss(y_true, qf, ef, theta)
                    all_fz_loss.append(fz)
            
            # Time
            for a, w, t in model_data['times']:
                if a == asset and w == window_idx:
                    all_times.append(t)
                    break
        
        summary['models'][model_name] = {
            'violation_rate_mean': np.mean(all_violations) if all_violations else np.nan,
            'violation_rate_std': np.std(all_violations) if all_violations else np.nan,
            'tick_loss_mean': np.mean(all_tick_loss) if all_tick_loss else np.nan,
            'tick_loss_std': np.std(all_tick_loss) if all_tick_loss else np.nan,
            'fz_loss_mean': np.mean(all_fz_loss) if all_fz_loss else np.nan,
            'fz_loss_std': np.std(all_fz_loss) if all_fz_loss else np.nan,
            'time_mean': np.mean(all_times) if all_times else np.nan,
            'n_experiments': len(all_violations)
        }
    
    return summary

#%% Main Execution

if __name__ == '__main__':
    print("\n" + "="*60)
    print("HAR-CAESar Comparative Experiment")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Data: {DATA_PATH}")
    print(f"  - Output: {OUTPUT_PATH}")
    print(f"  - Train window: {TRAIN_WINDOW} obs")
    print(f"  - Test window: {TEST_WINDOW} obs")
    print(f"  - Theta levels: {THETA_LEVELS}")
    print(f"  - Models: {MODELS}")
    
    all_results = {}
    all_summaries = {}
    
    for theta in THETA_LEVELS:
        # Run experiment
        results = run_experiment(theta)
        all_results[theta] = results
        
        # Compute summary
        summary = compute_summary(results)
        all_summaries[theta] = summary
        
        # Print summary
        print(f"\n\nSummary for theta = {theta}:")
        print("-" * 60)
        print(f"{'Model':<15} {'Viol.Rate':<12} {'Tick Loss':<12} {'FZ Loss':<12} {'Time(s)':<10}")
        print("-" * 60)
        
        for model_name in MODELS:
            if model_name in summary['models']:
                s = summary['models'][model_name]
                print(f"{model_name:<15} "
                      f"{s['violation_rate_mean']:.4f}       "
                      f"{s['tick_loss_mean']:.6f}    "
                      f"{s['fz_loss_mean']:.4f}      "
                      f"{s['time_mean']:.1f}")
    
    # Save final summary
    summary_path = f'{OUTPUT_PATH}/summary.pickle'
    with open(summary_path, 'wb') as f:
        pickle.dump(all_summaries, f)
    print(f"\n\nFinal summary saved to {summary_path}")
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)
