Theoretical Framework
=====================

The **HAR-CAESar** model combines the robust tail risk forecasting of CAESar (Conditional Autoregressive Expected Shortfall) with the long-memory capabilities of the HAR (Heterogeneous Autoregressive) volatility model.

The HAR-CAESar Model
--------------------

The core innovation is the modification of the autoregressive dynamics to condition on returns aggregated over different time horizons, inspired by the Heterogeneous Market Hypothesis (Muller et al., 1997).

Specification
^^^^^^^^^^^^^

The joint dynamics for Value at Risk ($Q_t$) and Expected Shortfall ($ES_t$) at probability level $\theta$ are given by:

.. math::

    Q_t &= \beta_0 + \beta_d r_{t-1}^{(d)} + \beta_w r_{t-1}^{(w)} + \beta_m r_{t-1}^{(m)} + \beta_q Q_{t-1} + \beta_e ES_{t-1} \\
    ES_t &= \gamma_0 + \gamma_d r_{t-1}^{(d)} + \gamma_w r_{t-1}^{(w)} + \gamma_m r_{t-1}^{(m)} + \gamma_q Q_{t-1} + \gamma_e ES_{t-1}

Where the heterogeneous return components are defined as:

*   **Daily:** $r_{t-1}^{(d)} = r_{t-1}$
*   **Weekly:** $r_{t-1}^{(w)} = \frac{1}{5} \sum_{j=1}^{5} r_{t-j}$
*   **Monthly:** $r_{t-1}^{(m)} = \frac{1}{22} \sum_{j=1}^{22} r_{t-j}$

Asymmetric Slope
^^^^^^^^^^^^^^^^

To capture leverage effects (where negative returns have a stronger impact on volatility), the model supports an **Asymmetric Slope (AS)** specification. Each return component is split into positive and negative parts:

.. math::

    \beta_h r_{t-1}^{(h)} \rightarrow \beta_h^+ (r_{t-1}^{(h)})^+ + \beta_h^- (r_{t-1}^{(h)})^-

Estimation Strategy
-------------------

The model is estimated using a multi-stage approach to ensure stability and consistency:

1.  **VaR Initialization:** Estimate Quantile Regression parameters using the Tick Loss.
2.  **ES Initialization:** Estimate Expected Shortfall parameters conditional on VaR.
3.  **Joint Optimization:** Re-estimate all parameters simultaneously by minimizing the joint **Fissler-Ziegel Loss** function:

.. math::

    L_{FZ}(r_t, Q_t, ES_t) = \frac{1}{T} \sum_{t=1}^T \left[ \frac{1}{\theta ES_t} (r_t - Q_t) \mathbf{1}_{\{r_t \le Q_t\}} + \frac{Q_t}{ES_t} + \log(-ES_t) - 1 \right]

Constraints are applied to ensure monotonicity ($ES_t \le Q_t$).
