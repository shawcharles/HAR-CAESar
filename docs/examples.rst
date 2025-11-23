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

Running the Full Experiment
---------------------------

To reproduce the empirical results from the thesis (comparing against benchmarks on real market data), use the provided experiment script:

.. code-block:: bash

    python experiments/experiments_har_caesar.py

This script will:
1.  Load data for S&P 500, FTSE 100, Nikkei 225, and MSCI EM.
2.  Run a rolling-window analysis.
3.  Save pickled results to ``output/har_experiment/``.
