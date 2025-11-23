import pickle
import numpy as np

# We check both 2.5% and 1.0% files
FILES = {
    '2.5%': 'CAESar/output/har_experiment/results_theta025.pickle',
    '1.0%': 'CAESar/output/har_experiment/results_theta001.pickle'
}

print("Capital Efficiency Analysis (Average Expected Shortfall)")
print("-" * 60)
print(f"{'Theta':<10} {'Model':<15} {'Avg ES (%)':<15} {'Capital Increase'}")
print("-" * 60)

for theta_label, path in FILES.items():
    try:
        with open(path, 'rb') as f:
            res = pickle.load(f)
        
        # Collect all ES forecasts for CAESar and HAR-CAESar
        es_caesar = []
        es_har = []
        
        # Iterate over all (asset, window, forecast) tuples
        for _, _, e in res['models']['CAESar']['ef']:
            es_caesar.extend(e)
            
        for _, _, e in res['models']['HAR_CAESar']['ef']:
            es_har.extend(e)
            
        avg_c = np.mean(es_caesar)
        avg_h = np.mean(es_har)
        
        # Capital held is proportional to |ES|
        # Increase = (|HAR| - |CAESar|) / |CAESar|
        incr = (abs(avg_h) - abs(avg_c)) / abs(avg_c)
        
        print(f"{theta_label:<10} {'CAESar':<15} {avg_c*100:6.4f}%")
        print(f"{'':<10} {'HAR-CAESar':<15} {avg_h*100:6.4f}%       {incr*100:+.2f}%")
        print("-" * 60)
        
    except FileNotFoundError:
        print(f"File not found: {path}")
