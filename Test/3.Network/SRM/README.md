# Social Relations Model (SRM) Analysis

This directory contains benchmarking and posterior validation tests for the Social Relations Model (SRM), a classic multivariate network model. It compares the performance (fitting time) and posterior distribution accuracy of **BayesForge (BF)** against **Stan**.

## Requirements

The required Python packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

*(Note: You must also have CmdStan installed for `cmdstanpy` to function properly, and R installed for `rpy2`.)*

## How to Run the Models

To execute the SRM benchmarking and comparison pipeline:

```bash
python fit_comparison.py
```

This script will simulate dyadic interaction data, fit the model using BayesForge, and subsequently fit the equivalent model using Stan, recording execution times and posterior samples.

## Outputs

The script generates the following outputs:
- `benchmark_results.csv`: Execution times and performance comparisons between the backends.
- `forest_plot*.png`: Visual plots comparing the posterior means and intervals of nodal random effects (sender/receiver) and dyadic effects across the backends.
- `log.txt`: Console outputs and sampling metrics.
