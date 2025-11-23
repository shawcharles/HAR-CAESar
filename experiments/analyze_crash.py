import pickle
import numpy as np
import pandas as pd
import datetime

# Load results for theta = 1% (extreme tail is most interesting for crashes)
RESULTS_PATH = 'CAESar/output/har_experiment/results_theta001.pickle'

with open(RESULTS_PATH, 'rb') as f:
    results = pickle.load(f)

# We want S&P 500 ('SP500')
# We need to find which window covers March 2020.
# The experiment saved 'windows' info.

target_asset = 'SP500'
crash_start = pd.Timestamp('2020-01-01')
crash_end = pd.Timestamp('2020-06-30')

found_data = False

print("Searching for COVID-19 window...")

for window_info in results['windows']:
    test_start = window_info['test_start']
    test_end = window_info['test_end']
    
    # Check if March 2020 is inside this test window
    if test_start <= pd.Timestamp('2020-03-01') and test_end >= pd.Timestamp('2020-03-31'):
        print(f"Found Window covering COVID: {test_start.date()} to {test_end.date()}")
        
        # Find the index of this window in the results list
        # We need to iterate through the model results to find the matching window index
        # This is a bit tricky because the results structure is list-of-tuples per model
        
        # Let's find the window index 'w_idx' corresponding to this timeframe
        # In the experiment script: results['windows'] is a list. The index in this list corresponds to window_idx.
        w_idx = results['windows'].index(window_info)
        
        # Extract data for this window and asset
        # Models: CAESar, HAR_CAESar
        
        y_true = None
        q_caesar = None
        e_caesar = None
        q_har = None
        e_har = None
        
        # Get True Returns
        for asset, w, y in results['models']['CAESar']['y_true']:
            if asset == target_asset and w == w_idx:
                y_true = y
                break
                
        # Get CAESar Forecasts
        for asset, w, q in results['models']['CAESar']['qf']:
            if asset == target_asset and w == w_idx:
                q_caesar = q
                break
        for asset, w, e in results['models']['CAESar']['ef']:
            if asset == target_asset and w == w_idx:
                e_caesar = e
                break
                
        # Get HAR-CAESar Forecasts
        for asset, w, q in results['models']['HAR_CAESar']['qf']:
            if asset == target_asset and w == w_idx:
                q_har = q
                break
        for asset, w, e in results['models']['HAR_CAESar']['ef']:
            if asset == target_asset and w == w_idx:
                e_har = e
                break
        
        # Create a DataFrame for analysis
        # We need dates.
        # The window_info gives start/end, but we need the specific dates for the test set.
        # We can reconstruct specific dates if we had the original index, 
        # but we can approximate or just print values around the max loss.
        
        if y_true is not None:
            # Find worst day
            worst_idx = np.argmin(y_true)
            worst_return = y_true[worst_idx]
            
            print(f"\nWorst Return in Window: {worst_return*100:.2f}% (Index {worst_idx})")
            print("-" * 50)
            print(f"{'Day':<5} {'Return':<10} {'CAESar VaR':<12} {'HAR VaR':<12} {'CAESar ES':<12} {'HAR ES':<12}")
            print("-" * 50)
            
            # Print +/- 5 days around the crash
            start_print = max(0, worst_idx - 5)
            end_print = min(len(y_true), worst_idx + 10)
            
            for i in range(start_print, end_print):
                r = y_true[i]
                qc = q_caesar[i]
                qh = q_har[i]
                ec = e_caesar[i]
                eh = e_har[i]
                
                marker = "<<" if r < qc else ""
                print(f"{i:<5} {r*100:6.2f}%   {qc*100:6.2f}%      {qh*100:6.2f}%      {ec*100:6.2f}%      {eh*100:6.2f}%   {marker}")
            
            # Calculate reaction speed
            # Mean VaR before crash vs Mean VaR after crash
            pre_crash = np.mean(q_caesar[:worst_idx])
            post_crash = np.mean(q_caesar[worst_idx:worst_idx+20])
            print(f"\nCAESar Reaction: {pre_crash*100:.2f}% -> {post_crash*100:.2f}%")
            
            pre_crash_h = np.mean(q_har[:worst_idx])
            post_crash_h = np.mean(q_har[worst_idx:worst_idx+20])
            print(f"HAR Reaction:    {pre_crash_h*100:.2f}% -> {post_crash_h*100:.2f}%")

        found_data = True
        break

if not found_data:
    print("Could not find a window covering March 2020 in the results.")
