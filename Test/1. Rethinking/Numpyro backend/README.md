# Rethinking Models - Numpyro Backend

This directory contains test models for the Numpyro backend. You can run all 13 models in sequence using the `run_all.py` script located in the `Numpyro backend/` folder. Running this script generates an output folder where, for each model, you will find density plot comparisons between BF and Stan for all parameters. Additionally, it outputs CSV files containing the posterior distributions, as well as the WAIC and LOO computation differences between BF and ArviZ.

## How to run

Navigate to the `Numpyro backend` directory or run the script directly by pointing your python interpreter to it.

### Basic Usage

``` bash
python "Numpyro backend/run_all.py"
```

### Controlling the Number of Simulations

You can control the number of simulations using the `BF_NSIM` environment variable. If not set, it defaults to `10`.

``` bash
# Run with 100 simulations
BF_NSIM=100 python "Numpyro backend/run_all.py"
```

### Available Arguments

The script accepts the following command-line arguments:

-   `--shard` : Enables data sharding across CPU cores (`shard=True` in fit()). When this flag is passed, logs and plots are saved in separate locations (`log_shard.txt` and a `plots_shard/` directory) to avoid overwriting standard runs.
-   `--start <N>` : Starts the execution from model number `N` (1-indexed). The default is `1`. For example, `--start 5` skips the first 4 models and starts from the 5th model.

### Examples

**Run with sharding enabled and 500 simulations:**

``` bash
BF_NSIM=100 python "Numpyro backend/run_all.py" --shard
```

**Resume execution starting from the 3rd model:**

``` bash
python "Numpyro backend/run_all.py" --start 3
```