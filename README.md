# HAR-CAESar

[![Tests](https://github.com/shawcharles/HAR-CAESar/actions/workflows/tests.yml/badge.svg)](https://github.com/shawcharles/HAR-CAESar/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**HAR-CAESar** (Heterogeneous Autoregressive Conditional Autoregressive Expected Shortfall) is a Python library for joint forecasting of Value at Risk (VaR) and Expected Shortfall (ES). The full thesis is available [here](thesis.pdf).

Developed for the MSc Thesis **"Forecasting Tail Risk with Long-Memory"**, this model extends the CAESar framework by incorporating heterogeneous volatility components (daily, weekly, monthly) to better capture long-memory dynamics in financial tail risk.

## Key Features

-   **Long-Memory Modeling**. Captures volatility cascades using HAR dynamics (Corsi, 2009).
-   **Asymmetric Slope Effects**. Separates positive and negative return impacts at daily, weekly, and monthly horizons.
-   **Joint Estimation**. Simultaneous VaR and ES estimation using the Fissler-Ziegel consistent loss function.
-   **Robust Estimation**. Three-stage procedure with multiple random initializations and convergence verification.
-   **Comprehensive Backtesting**. Implements Kupiec, Christoffersen, McNeil-Frey, and Acerbi-Szekely tests.
-   **Benchmarks**. Includes CAESar, CAViaR, GAS1, and GAS2 implementations for comparison.

## Model Specification

The HAR-CAESar model modifies the autoregressive structure of tail risk to condition on multi-horizon returns:

```math
\text{VaR}_t = \beta_0 + \beta_d r_{t-1}^{(d)} + \beta_w r_{t-1}^{(w)} + \beta_m r_{t-1}^{(m)} + \beta_q \text{VaR}_{t-1} + \beta_e \text{ES}_{t-1}
```

Where $r^{(d)}$, $r^{(w)}$, and $r^{(m)}$ represent daily, weekly, and monthly aggregated return components, allowing the model to adapt to different frequency information flows.

## Installation

```bash
git clone https://github.com/shawcharles/HAR-CAESar.git
cd HAR-CAESar
pip install -e .
```

## Quick Start

Run the full experimental pipeline to reproduce thesis results:

```bash
python experiments/experiments_har_caesar.py
```

Results will be saved to `output/har_experiment/`.

## Usage Example

```python
import numpy as np
from har_caesar import HAR_CAESar

# 1. Generate synthetic data
y = np.random.normal(0, 1, 2000)
theta = 0.025  # 2.5% tail probability

# 2. Initialize model
model = HAR_CAESar(theta=theta)

# 3. Fit and Predict
# Split at index 1500: fit on [0:1500], predict on [1500:end]
results = model.fit_predict(y, ti=1500, seed=42)

# 4. Analyze Results
print(f"Mean VaR Forecast: {np.mean(results['qf']):.4f}")
print(f"Mean ES Forecast:  {np.mean(results['ef']):.4f}")
```

## Repository Structure

-   `src/har_caesar/`: Main package source code.
-   `experiments/`: Reproducible scripts for empirical analysis.
-   `tests/`: Unit tests for model verification.
-   `data/`: Input dataset location.
-   `output/`: Generated results and artifacts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite the thesis:

> Shaw, C. (2025). *Forecasting Tail Risk with Long-Memory: A Heterogeneous Extension of the CAESar Model* (MSc Thesis).

## Acknowledgments

This work builds upon the original CAESar implementation by [Federico Gatta](https://github.com/fgt996/CAESar).
