# EWA (Experience-Weighted Attraction) Benchmark

This folder contains a benchmark script for comparing the **Experience-Weighted Attraction (EWA)** learning model implemented in Stan against its equivalent in **BayesForge**.

The script tests a cognitive model applied to foraging data (the `panama_ewa` dataset), comparing how individual monkeys learn techniques based on personal yields and 5 different social cues (frequency, payoff bias, kin bias, rank bias, cohort bias, and age bias).

## Requirements

To run this benchmark, ensure you have the following Python packages installed: 
- `BayesForge` 
- `cmdstanpy` (and a working CmdStan installation)
- `numpy` 
- `pandas`  
- `matplotlib` 
-  `seaborn`

You can install the dependencies (excluding BayesForge if already installed) via:

``` bash
pip install cmdstanpy numpy pandas matplotlib seaborn
```

## How to Run the Models

A `run_comparison.py` script is provided to automate the benchmark. It will: 
1. Load and format the foraging data. 
2. Compile and sample the Stan model (`Stan/PN_social_global_age.stan`). 
3. Build and sample the BayesForge model (`BF_backend/model.py`). 
4. Compare the posterior estimates between both engines.

To run the pipeline, simply execute:

``` bash
python3 run_comparison.py
```

## Outputs

The script will generate the following outputs in the root of the folder: 
- `log.txt`: A text file containing a table of the posterior means for the key parameters (`lambda`, `mu`, `sigma`, `b_age`) from both Stan and BayesForge, along with the difference between the two estimates. 
- `density_comparison.png`: A visual grid of kernel density estimate (KDE) plots comparing the full posterior distributions (Stan in blue vs. BayesForge in orange) for all tracked parameters.