# BNN Tests Directory (10.BNN)

This folder contains testing and diagnostic scripts for evaluating the Bayesian Neural Network (BNN) capabilities in the `BayesForge` library.

## What it does

The scripts in this directory test the performance and correctness of BNN models: - **`BNN covariance.py`**: Tests the estimation of covariance matrices in a multi-level modeling context (e.g., varying intercepts and slopes for cafe wait times) using a BNN layer (`m.bnn.cov`) and compares it against a standard multi-level model. - **`BNN regressions.py`**: Tests BNN regressions with a two-hidden-layer network (`m.bnn.layer_linear`) on both linear and non-linear synthetic data. It validates the network's ability to model functions and capture uncertainty (Credible Intervals). - **`9.BNN.ipynb`**: A Jupyter Notebook containing explorations or interactive versions of the BNN tests.

## How to run the tests

You can run all the `.py` tests sequentially using the provided `run_all.py` script:

``` bash
python run_all.py
```

Alternatively, you can run each script individually:

``` bash
python "BNN covariance.py"
python "BNN regressions.py"
```

## What they produce

Executing the scripts will fit the respective models, print diagnostic metrics (like the recovered correlation/rho matrix compared to the original), and generate the following output files: - **`BNN_cov.png`**: A scatter plot comparing the posterior distributions from the standard model and the BNN model (produced by `BNN covariance.py`). - **`bnn_prediction_linear.png`**: A plot showing the BNN mean predictions with 90% credible intervals against linear synthetic training data (produced by `BNN regressions.py`). - **`bnn_prediction_non_linear.png`**: A plot showing the BNN mean predictions with 90% credible intervals against non-linear synthetic training data (produced by `BNN regressions.py`). - **`BNN.json`**: A JSON file containing the standardized non-linear dataset (X and Y coordinates) used in the regression test (produced by `BNN regressions.py`).