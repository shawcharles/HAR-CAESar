Utility Functions
=================

The ``har_caesar.utils`` module provides essential statistical testing and loss functions for model validation and comparison.

Loss Functions
--------------

The module includes implementations of elicitable loss functions for VaR and ES:

*   **Tick Loss:** Quantile loss for VaR estimation
*   **Barrera Loss:** Loss function for ES residual estimation
*   **Fissler-Ziegel Loss:** Jointly consistent loss for (VaR, ES) pairs

Statistical Tests
-----------------

**VaR Backtests:**

*   ``kupiec_test()``: Unconditional coverage test
*   ``christoffersen_cc_test()``: Conditional coverage test (UC + independence)

**ES Backtests:**

*   ``mcneil_frey_test()``: Bootstrap-based ES calibration test
*   ``acerbi_szekely_test()``: Z1 and Z2 tests for ES specification

**Forecast Comparison:**

*   ``diebold_mariano_test()``: HAC-robust test for comparing forecast accuracy
*   ``bootstrap_loss_test()``: One-sided test for forecast encompassing

API Documentation
-----------------

.. automodule:: har_caesar.utils
   :members:
   :undoc-members:
   :special-members: __call__
   :show-inheritance:
