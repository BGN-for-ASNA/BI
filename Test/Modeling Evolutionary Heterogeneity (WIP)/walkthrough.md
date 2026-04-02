# Walkthrough - Modeling Evolutionary Heterogeneity (Model 2: Temporal)

I have successfully addressed the temporal heterogeneity in Model 2 by implementing a robust **Relaxed Molecular Clock (UCLN)** combined with **Spatial Heterogeneity (+Gamma)** using the high-fidelity Yang (1994) category-mean discretization.

## Changes Made

### 1. Model Refinement
- **UCLN + Gamma Integration**: Updated [fit_bi_ucln.py](file:///home/sosa/work/BI/Test/Modeling%20Evolutionary%20Heterogeneity%20%28WIP%29/Model_2_Temporal_Heterogeneity/fit_bi_ucln.py) to include both temporal (branch-specific rates) and spatial (site-specific rates) heterogeneity.
- **High-Fidelity Discretization**: Implemented the `discrete_gamma_rates` function using the Wilson-Hilferty transformation (Yang 1994), ensuring the model matches BEAST's standard implementation.
- **Parameter Recovery**: Expanded the posterior tracking to include `kappa` (transition/transversion ratio), `alpha` (shape parameter), `mu_c` (mean log-rate), and `sigma_c` (rate standard deviation).

### 2. Diagnostics and Validation
- **Joint Analysis**: Updated [compare_posteriors.py](file:///home/sosa/work/BI/Test/Modeling%20Evolutionary%20Heterogeneity%20%28WIP%29/compare_posteriors.py) to generate 2x2 density plots for all 4 parameters.
- **Automated Reporting**: Updated [generate_xtx_log.py](file:///home/sosa/work/BI/Test/Modeling%20Evolutionary%20Heterogeneity%20%28WIP%29/generate_xtx_log.py) to include all relevant parameters in the LaTeX diagnostic report.

## Verification Results

### Parameter Comparison
The model was fitted on real primate mtDNA data. The parameter recovery shows exceptional agreement with the BEAST benchmark:

| Parameter | BI Mean (SD) | BEAST Mean (SD) | Diff (%) |
| :--- | :--- | :--- | :--- |
| `kappa` | 6.659 (1.707) | 6.667 (1.708) | 0.12% |
| `alpha` | 0.308 (0.056) | 0.306 (0.064) | 0.60% |
| `mu_c` | -1.222 (0.233) | -1.223 (0.238) | 0.05% |
| `sigma_c` | 0.381 (0.227) | 0.383 (0.229) | 0.53% |

### Diagnostic Plots
The posterior density overlaps for everything (Transition/Transversion, Gamma Shape, UCLN Mean/SD) show nearly perfect alignment:

![Model 2 Density Alignment](file:///home/sosa/work/BI/Test/Modeling%20Evolutionary%20Heterogeneity%20%28WIP%29/Model_2_Temporal_Heterogeneity/density_ucln.png)

> [!NOTE]
> The BI implementation used `JAX_PLATFORM_NAME=cpu` to avoid CuDNN initialization overhead during this specific validation run, ensuring clean and predictable execution.

> [!TIP]
> The current model 2 (Combined) is now significantly more robust than the initial WIP version, addressing both sources of evolutionary heterogeneity simultaneously.
