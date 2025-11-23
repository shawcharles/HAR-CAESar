.. HAR-CAESar documentation master file

Welcome to HAR-CAESar's documentation!
======================================

**HAR-CAESar** is a Python framework for forecasting joint **Value at Risk (VaR)** and **Expected Shortfall (ES)** using heterogeneous autoregressive dynamics.

This project was developed for the MSc Thesis in Applied Statistics: *"Forecasting Tail Risk with Long-Memory: A Heterogeneous Extension of the CAESar Model"*.

Project Overview
----------------

The library extends the **CAESar** model (Gatta, Lillo, & Mazzarisi, 2024) by integrating **HAR** components (Corsi, 2009). This allows the model to capture long-memory volatility cascades by conditioning tail risk on daily, weekly, and monthly aggregated returns.

Key Features
------------

*   **HAR-CAESar Model:** A novel specification that captures multi-horizon volatility dynamics.
*   **Joint Estimation:** Estimates VaR and ES simultaneously using the Fissler-Ziegel loss function.
*   **Robust Backtesting:** Includes tools for uncoditional coverage, conditional coverage, and ES calibration tests.
*   **Comparison Benchmarks:** Includes implementations of standard CAESar, CAViaR, and GAS models.

Installation
------------

Create the environment using Conda:

.. code-block:: bash

   conda env create -f CAESar_env.yml
   conda activate CAESar

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from models.har_caesar import HAR_CAESar

   # Generate dummy data
   y = np.random.normal(0, 1, 1000)
   tv = 800 # Split point

   # Initialize and fit
   mdl = HAR_CAESar(theta=0.025, specification='AS')
   res = mdl.fit_predict(y, tv, seed=42)

   print("VaR Forecasts:", res['qf'])

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
