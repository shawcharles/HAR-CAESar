.. HAR-CAESar documentation master file

HAR-CAESar Documentation
========================

**HAR-CAESar** (Heterogeneous Autoregressive Conditional Autoregressive Expected Shortfall) is a Python framework for forecasting joint **Value at Risk (VaR)** and **Expected Shortfall (ES)**.

This library implements the models developed for the MSc Thesis: *"Forecasting Tail Risk with Long-Memory: A Heterogeneous Extension of the CAESar Model"*. _.

Key Features
------------

* **Long-Memory Modeling**: Captures volatility cascades using HAR dynamics at daily, weekly, and monthly horizons
* **Asymmetric Slope Effects**: Separates positive and negative return impacts to capture leverage effects
* **Joint Estimation**: Simultaneous VaR and ES forecasting using the Fissler-Ziegel loss function
* **Comprehensive Backtesting**: Implements Kupiec, Christoffersen, McNeil-Frey, and Acerbi-Szekely tests
* **Benchmarks**: Includes CAESar, CAViaR, GAS1, and GAS2 implementations for comparison

Quick Start
-----------

Install the package:

.. code-block:: bash

   git clone https://github.com/shawcharles/HAR-CAESar.git
   cd HAR-CAESar
   pip install -e .

Basic usage example:

.. code-block:: python

   import numpy as np
   from har_caesar import HAR_CAESar

   # Generate data
   y = np.random.normal(0, 1, 2000)
   
   # Fit model and predict
   model = HAR_CAESar(theta=0.025)
   results = model.fit_predict(y, ti=1500, seed=42)
   
   # Access forecasts
   var_forecasts = results['qf']
   es_forecasts = results['ef']

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   theory
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
