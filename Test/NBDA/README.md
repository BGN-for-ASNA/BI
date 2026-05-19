# NBDA Model Comparison

Compares NBDA models implemented in the **BI** (BayesianInference) framework against their
[STbayes](https://github.com/michaelchimento/STbayes) equivalents using overlapping posterior
density plots and benchmark timing.

## Prerequisites

Install STbayes from CRAN or GitHub before running any script:

```r
# From GitHub
remotes::install_github("michaelchimento/STbayes")
```

`BayesianInference` (the BI R package) and its Python backend must also be installed.

## Directory Structure

```
BI_Models/          Core BI model definitions (one file per model)
density_plots/      Output PNG files (generated, gitignored)
density_plots.R     Main script — fits all 8 models, produces overlapping density plots
                    with symmetric KL divergence annotations per parameter
full_comparison.R   Benchmark script — timing comparison BI vs STbayes for all 8 models
install_deps.R      R dependency installer
```

## Running

### Density comparison plots

```r
setwd("Test/NBDA")
source("density_plots.R")
# Output: density_plots/<model>.png
```

### Full benchmark

```r
setwd("Test/NBDA")
source("full_comparison.R")
```

## Models Covered

| Model | Description |
|-------|-------------|
| cTADA | Continuous Time of Acquisition |
| OADA | Order of Acquisition |
| OADA_asocial | Order of Acquisition, asocial only |
| ILV | Individual-Level Variables |
| veff | Varying Effects (hierarchical random effects) |
| dynamic_tweights | Dynamic Networks with time-varying weights |
| complex_f | Frequency-dependent transmission |
| posterior_edges | Network uncertainty via posterior edge weights |

## Reading the density plots

Each panel shows the marginal posterior for one parameter.
- **Blue** = STbayes (Stan/NUTS)
- **Orange** = BI (JAX/NUTS)
- **sym-KL** = symmetric KL divergence between the two posteriors (lower = better agreement)
- Panels labelled **[BI only]** or **[STb only]** indicate that the parameter has no
  equivalent in the other framework (or the other fit failed).

## Known differences

### veff — varying effects not applied to `s_prime` in STbayes

When calling `generate_STb_model(..., veff_params = c("lambda_0", "s_prime"))`, the
STbayes-generated Stan code allocates a 2-dimensional varying-effects structure
(`sigma_id[N_veff]`, `z_id[N_veff, P]`, LKJ correlation matrix) but only applies
the individual offset to `lambda_0` via `v_id[,1]`. The `s_prime` parameter remains
a population-level scalar (`s_prime = exp(log_s_prime_mean)`); `v_id[,2]` is never
used in the likelihood.

Consequence: `sigma_id[2]` is sampled from its prior with no likelihood information
(BI only in the veff plot), and `log_s_prime_mean` posteriors diverge (KL ≈ 0.36)
because BI correctly absorbs inter-individual variance into per-individual `s_prime`
offsets while STbayes cannot.

This appears to be an incomplete implementation in the STbayes code generator rather
than an intentional design choice (allocating an unused parameter wastes MCMC samples).
Worth raising as an issue at https://github.com/michaelchimento/STbayes.
