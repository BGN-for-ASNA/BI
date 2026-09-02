# BayesForge Distribution Tests

This folder contains testing scripts to verify that various probability distributions provided by the `BayesForge` library (`bf.dist.*`) instantiate correctly without throwing errors in their respective JAX backends.

## Scripts

-   **`distribution_numpyro.py`**: Tests the distribution implementations using the default **NumPyro** backend.
-   **`distribution_tfp.py`**: Tests the distribution implementations using the **TensorFlow Probability (TFP)** backend.

*Note: In `distribution_tfp.py`, a few distributions are currently commented out because they are either not supported by the TFP JAX substrate (e.g. `AsymmetricLaplace`, `TruncatedDistribution`) or have conflicting parameters in the `tfp_dists.py` wrapper (e.g. `GammaPoisson`).*

## How to Run

1.  Make sure your virtual environment is activated

2.  Run the scripts using Python:

    ``` bash
    python distribution_numpyro.py
    python distribution_tfp.py
    ```

## Logs & Outputs

Instead of printing errors to the terminal, both scripts capture instantiation errors and output them to plain text log files in this same directory: - **`log_numpyro.txt`**: Records any distributions that failed to instantiate in the NumPyro backend, along with their stack trace/error messages. If all pass, the log file may just contain an empty list `[]` or no errors. - **`log_tfp.txt`**: Records any distributions that failed to instantiate in the TFP backend. The file will list the distribution name followed by the exact error message that was thrown.