Theoretical Framework
=====================

The **HAR-CAESar** model combines the robust tail risk forecasting of CAESar (Conditional Autoregressive Expected Shortfall) with the long-memory capabilities of the HAR (Heterogeneous Autoregressive) volatility model.

The HAR-CAESar Model
--------------------

The core innovation is the modification of the autoregressive dynamics to condition on returns aggregated over different time horizons, inspired by the Heterogeneous Market Hypothesis (Muller et al., 1997).

Specification
^^^^^^^^^^^^^

The joint dynamics for Value at Risk ($Q_t$) and Expected Shortfall ($ES_t$) at probability level $\theta$ use an **Asymmetric Slope (AS)** structure at all horizons to capture leverage effects:

**VaR Equation:**

.. math::

    Q_t = \beta_0 + \beta_1^{(d)} (r_{t-1}^{(d)})^+ + \beta_2^{(d)} (r_{t-1}^{(d)})^- + \beta_1^{(w)} (r_{t-1}^{(w)})^+ + \beta_2^{(w)} (r_{t-1}^{(w)})^- + \beta_1^{(m)} (r_{t-1}^{(m)})^+ + \beta_2^{(m)} (r_{t-1}^{(m)})^- + \beta_3 Q_{t-1} + \beta_4 ES_{t-1}

**ES Equation:**

.. math::

    ES_t = \gamma_0 + \gamma_1^{(d)} (r_{t-1}^{(d)})^+ + \gamma_2^{(d)} (r_{t-1}^{(d)})^- + \gamma_1^{(w)} (r_{t-1}^{(w)})^+ + \gamma_2^{(w)} (r_{t-1}^{(w)})^- + \gamma_1^{(m)} (r_{t-1}^{(m)})^+ + \gamma_2^{(m)} (r_{t-1}^{(m)})^- + \gamma_3 Q_{t-1} + \gamma_4 ES_{t-1}

Where:

*   **Daily:** $r_{t-1}^{(d)} = r_{t-1}$
*   **Weekly:** $r_{t-1}^{(w)} = \frac{1}{5} \sum_{j=1}^{5} r_{t-j}$
*   **Monthly:** $r_{t-1}^{(m)} = \frac{1}{22} \sum_{j=1}^{22} r_{t-j}$
*   **Positive/Negative:** $(x)^+ = \max(0, x)$ and $(x)^- = \max(0, -x)$

This specification has **9 parameters per equation** (intercept + 6 return coefficients + 2 autoregressive terms), allowing positive and negative returns to have different impacts at each horizon.

Estimation Strategy
-------------------

The model is estimated using a three-stage approach to ensure stability and consistency:

**Stage 1: VaR Initialization**

Estimate VaR parameters using CAViaR with the Tick (quantile) Loss:

.. math::

    L_{tick}(r_t, Q_t) = (r_t - Q_t)(\theta - \mathbf{1}_{\{r_t < Q_t\}})

**Stage 2: ES Residual Estimation**

Rather than estimating ES directly, Stage 2 estimates the **ES residual** $r_t = ES_t - Q_t$ using the Barrera loss function. This reparametrisation ensures monotonicity ($ES_t \le Q_t < 0$) since $r_t < 0$.

**Stage 3: Joint Refinement**

Re-estimate all parameters simultaneously by minimizing the joint **Fissler-Ziegel Loss** function:

.. math::

    L_{FZ}(r_t, Q_t, ES_t) = \frac{1}{T} \sum_{t=1}^T \left[ \frac{1}{\theta ES_t} (r_t - Q_t) \mathbf{1}_{\{r_t \le Q_t\}} + \frac{Q_t}{ES_t} + \log(-ES_t) - 1 \right]

**Implementation Details:**

*   Penalty weights: $\lambda_q = \lambda_e = 10$ enforce the monotonicity constraint $ES_t \le Q_t$
*   Multiple random initializations ensure global convergence
*   Convergence verified using gradient tolerance and loss function stability

Backtesting
-----------

The implementation provides comprehensive backtesting tools:

**VaR Backtests:**

*   **Kupiec (1995):** Unconditional coverage test for correct violation rate
*   **Christoffersen (1998):** Conditional coverage test combining correct rate and independence

**ES Backtests:**

*   **McNeil-Frey (2000):** Bootstrap test for ES calibration using exceedance residuals
*   **Acerbi-Szekely (2014):** Z1 and Z2 tests for ES specification

**Forecast Comparison:**

*   **Diebold-Mariano:** HAC-robust test for predictive accuracy comparison
*   **Bootstrap loss differential:** One-sided test for forecast encompassing
