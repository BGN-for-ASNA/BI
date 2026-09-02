# Survival Analysis Verification

This directory contains a verification script to compare the output of the **BayesForge (BF)** survival analysis module with a standard **PyMC** implementation.

## Overview

The primary script is `Survival_analysis_verify.py`. It runs a Poisson survival model (specifically for the mastectomy dataset) in both BayesForge and PyMC to explicitly verify parameter consistency (baseline rates `lambda0` and hazard rate `beta`). 

## Requirements

Ensure you have the required packages installed in your environment:
```bash
pip install -r requirements.txt
```
*(Requires `pymc` and `BayesForge`, along with their standard dependencies).*

## Running the Verification

To run the verification model:
```bash
python Survival_analysis_verify.py
```

### Outputs

The script will automatically generate the following plots in this directory:
1. **`survival_comparison_verify.png`**: A KDE density plot comparing the posterior distribution of the `beta` (metastasized) coefficient between BF and PyMC.
2. **`survival_scatter_verify.png`**: A scatter plot comparing the mean posterior estimates of all parameters (`lambda0` baseline rates and the `beta` hazard rate) from both models against a $y = x$ reference line. 

It will also print a summary to the console comparing the mean beta values to prove numerical consistency!
