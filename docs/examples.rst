Examples
========

This section provides examples of how to use the HAR-CAESar library.

Basic Usage
-----------

The following example demonstrates how to generate synthetic data, fit the model, and produce forecasts.

.. code-block:: python

    import numpy as np
    from har_caesar import HAR_CAESar

    # 1. Generate synthetic returns (e.g., from a GARCH process)
    np.random.seed(42)
    y = np.random.normal(0, 1, 2000)
    theta = 0.025  # 2.5% tail probability

    # 2. Initialize the model
    # 'AS' (Asymmetric Slope) is the default specification
    model = HAR_CAESar(theta=theta, lambdas={'q': 10, 'e': 10})

    # 3. Fit and Predict
    # We split the data at index 1500:
    # - Train on y[:1500]
    # - Predict for y[1500:]
    print("Fitting model...")
    results = model.fit_predict(y, ti=1500, seed=42)

    # 4. Access Results
    q_forecasts = results['qf']  # VaR
    e_forecasts = results['ef']  # ES

    print(f"Forecasts generated: {len(q_forecasts)}")
    print(f"Mean VaR: {np.mean(q_forecasts):.4f}")

Backtesting Example
-------------------

This example demonstrates how to validate VaR forecasts using the Christoffersen conditional coverage test:

.. code-block:: python

    import numpy as np
    from har_caesar import HAR_CAESar
    from har_caesar.utils import christoffersen_cc_test

    # 1. Generate synthetic data
    np.random.seed(42)
    y = np.random.normal(0, 1, 2000)
    theta = 0.025  # 2.5% VaR

    # 2. Fit model and generate forecasts
    model = HAR_CAESar(theta=theta)
    results = model.fit_predict(y, ti=1500, seed=42)
    
    # 3. Get test period data
    y_test = y[1500:]
    var_forecasts = results['qf']
    
    # 4. Compute violations (1 if return < VaR, 0 otherwise)
    violations = (y_test < var_forecasts).astype(int)
    
    # 5. Run Christoffersen test
    test_results = christoffersen_cc_test(violations, theta=theta)
    
    # 6. Interpret results
    print(f"Violation rate: {np.mean(violations):.4f} (expected: {theta:.4f})")
    print(f"LR_UC: {test_results['LR_UC']:.2f}, p-value: {test_results['p_value_UC']:.4f}")
    print(f"LR_IND: {test_results['LR_IND']:.2f}, p-value: {test_results['p_value_IND']:.4f}")
    print(f"LR_CC: {test_results['LR_CC']:.2f}, p-value: {test_results['p_value_CC']:.4f}")
    
    if test_results['p_value_CC'] > 0.05:
        print("✓ Model passes Christoffersen test at 5% significance")
    else:
        print("✗ Model fails Christoffersen test at 5% significance")

Running the Full Experiment
---------------------------

To reproduce the empirical results from the thesis (comparing against benchmarks on real market data), use the provided experiment script:

.. code-block:: bash

    python experiments/experiments_har_caesar.py

This script will:
1.  Load data for S&P 500, FTSE 100, Nikkei 225, and MSCI EM.
2.  Run a rolling-window analysis.
3.  Save pickled results to ``output/har_experiment/``.
