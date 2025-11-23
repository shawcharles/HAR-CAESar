# HAR-CAESar: Heterogeneous Autoregressive Extension for CAESar

This repository contains the code and results for the MSc Thesis **"Forecasting Tail Risk with Long-Memory: A Heterogeneous Extension of the CAESar Model"**.

It extends the original **CAESar** (Conditional Autoregressive Expected Shortfall) framework proposed by Gatta, Lillo, and Mazzarisi (2024) by incorporating **Heterogeneous Autoregressive (HAR)** volatility components (Corsi, 2009) to capture long-memory dynamics in tail risk forecasting.

## Project Overview

The core contribution is the implementation of the `HAR_CAESar` class, which modifies the original autoregressive dynamics to condition on daily, weekly, and monthly aggregated returns.

*   **Original CAESar:** Conditions on $r_{t-1}$, $\text{VaR}_{t-1}$, $\text{ES}_{t-1}$.
*   **HAR-CAESar:** Conditions on $r_{t-1}^{(d)}$, $r_{t-1}^{(w)}$, $r_{t-1}^{(m)}$, $\text{VaR}_{t-1}$, $\text{ES}_{t-1}$.

## Repository Structure

*   **code/models/**
    *   `caesar.py`: Original CAESar implementation (Gatta et al., 2024).
    *   `har_caesar.py`: **NEW** HAR-CAESar implementation.
    *   `gas.py`: Generalised Autoregressive Score models (Patton et al., 2019).
    *   `caviar.py`: CAViaR implementation (Engle & Manganelli, 2004).
*   **code/experiments/**
    *   `har_experiment.py`: Main script for running the rolling-window comparison.
*   **output/har_experiment/**
    *   Contains the `.pickle` files with the results of the empirical analysis (S&P 500, FTSE 100, Nikkei 225, MSCI EM).

## Getting Started

### Prerequisites
Create the conda environment using the provided file:

```bash
conda env create -f CAESar_env.yml
conda activate CAESar
```

### Running the HAR-CAESar Model

The syntax for `HAR_CAESar` follows the same API as the original model.

```python
import numpy as np
from models.har_caesar import HAR_CAESar

# Example Data
y = np.random.normal(0, 1, 1000) # Log-returns
tv = 800 # Train/Test split point
theta = 0.025 # Probability level

# Initialize and Fit
mdl = HAR_CAESar(theta, 'AS') # 'AS' = Asymmetric Slope
res = mdl.fit_predict(y, tv, seed=42)

# Access Results
print("VaR Forecasts:", res['qf'])
print("ES Forecasts:", res['ef'])
```

### Reproducing Thesis Results

To reproduce the empirical analysis presented in the thesis:

1.  Navigate to the code directory:
    ```bash
    cd code
    ```
2.  Run the HAR experiment script:
    ```bash
    python har_experiment.py
    ```
    This will generate the results in `output/har_experiment/`.

3.  Analyze the results (Backtests & Loss Functions):
    ```bash
    python analyze_har_results.py
    ```

## References

*   **Original CAESar Paper:** Gatta, F., Lillo, F., & Mazzarisi, P. (2024). CAESar: Conditional Autoregressive Expected Shortfall. *arXiv preprint arXiv:2407.06619*.
*   **HAR Model:** Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*.
*   **Thesis:** *Forecasting Tail Risk with Long-Memory: A Heterogeneous Extension of the CAESar Model* (2025).

## Acknowledgments

This code is based on the original implementation by [Federico Gatta](https://github.com/fgt996/CAESar).
