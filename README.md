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

The HAR-CAESar model extends CAESar with heterogeneous autoregressive dynamics, incorporating asymmetric slope effects at multiple horizons:

**VaR Equation:**
```math
\text{VaR}_t = \beta_0 + \beta_1^{(d)} (r_{t-1}^{(d)})^+ + \beta_2^{(d)} (r_{t-1}^{(d)})^- + \beta_1^{(w)} (r_{t-1}^{(w)})^+ + \beta_2^{(w)} (r_{t-1}^{(w)})^- + \beta_1^{(m)} (r_{t-1}^{(m)})^+ + \beta_2^{(m)} (r_{t-1}^{(m)})^- + \beta_3 \text{VaR}_{t-1} + \beta_4 \text{ES}_{t-1}
```

**ES Equation:**
```math
\text{ES}_t = \gamma_0 + \gamma_1^{(d)} (r_{t-1}^{(d)})^+ + \gamma_2^{(d)} (r_{t-1}^{(d)})^- + \gamma_1^{(w)} (r_{t-1}^{(w)})^+ + \gamma_2^{(w)} (r_{t-1}^{(w)})^- + \gamma_1^{(m)} (r_{t-1}^{(m)})^+ + \gamma_2^{(m)} (r_{t-1}^{(m)})^- + \gamma_3 \text{VaR}_{t-1} + \gamma_4 \text{ES}_{t-1}
```

Where:
- $r^{(d)}$, $r^{(w)}$, $r^{(m)}$ = daily, weekly (5-day), monthly (22-day) aggregated returns
- $(x)^+ = \max(0, x)$ and $(x)^- = \max(0, -x)$ = positive and negative components
- **9 parameters per equation** (intercept + 6 return coefficients + 2 AR terms)

The asymmetric slope structure allows positive and negative returns to have different impacts at each horizon, while the multi-horizon components capture long-memory effects in tail risk.

### Estimation

The model uses a three-stage procedure:
1. **Stage 1**: CAViaR estimation for initial VaR
2. **Stage 2**: ES residual estimation (r = ES - VaR) using Barrera loss
3. **Stage 3**: Joint refinement with Fissler-Ziegel loss (penalty weights λ = 10)

## Statistical Testing

The package provides comprehensive backtesting tools for VaR and ES validation:

### VaR Backtests
- **Kupiec (1995)**: Unconditional coverage test (correct violation rate)
- **Christoffersen (1998)**: Conditional coverage test (correct rate + independence of violations)

### ES Backtests
- **McNeil-Frey (2000)**: Bootstrap test for ES calibration using exceedance residuals
- **Acerbi-Szekely (2014)**: Z1 and Z2 tests for ES specification

### Forecast Comparison
- **Diebold-Mariano**: HAC-robust test for predictive accuracy comparison
- **Bootstrap loss differential**: One-sided test for forecast encompassing

Example usage:
```python
from har_caesar.utils import christoffersen_cc_test

# Compute violations
violations = (y_test < var_forecasts).astype(int)

# Run Christoffersen test
results = christoffersen_cc_test(violations, theta=0.025)
print(f"LR_CC: {results['LR_CC']:.2f}, p-value: {results['p_value_CC']:.4f}")
```

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

-   `src/har_caesar/`: Main package source code
    -   `models/`: Model implementations (HAR-CAESar, CAESar, CAViaR, GAS)
    -   `utils.py`: Statistical testing and loss functions
-   `experiments/`: Reproducible scripts for empirical analysis
    -   `experiments_har_caesar.py`: Main comparative experiment
    -   `generate_synthetic_data.py`: Synthetic data generation
-   `tests/`: Unit tests for model verification
-   `docs/`: Sphinx documentation (API reference, theory, examples)
-   `memory-bank/`: Project context and development notes
-   `data/`: Input dataset location
-   `output/`: Generated results and artifacts

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite the thesis:

> Shaw, C. (2025). *Forecasting Tail Risk with Long-Memory: A Heterogeneous Extension of the CAESar Model* (MSc Thesis).

## Acknowledgments

This work builds upon the original CAESar implementation by [Federico Gatta](https://github.com/fgt996/CAESar).
