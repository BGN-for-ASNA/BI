# Trait coevolution and causal inference using generalized dynamic phylogenetic models

This repository contains data and code for reproducing the paper:

> Ringen, E. J., Claessens, S., Martin, J. S., & Jaeggi, A. V. (2026). Trait coevolution and causal inference using generalized dynamic phylogenetic models. *Methods in Ecology and Evolution*.

The `coevolve` R package and tutorials are available at: https://scottclaessens.github.io/coevolve/

## Repository structure

```
├── manuscript.qmd                # Main manuscript (Quarto)
├── supporting_information.qmd    # Supporting Information (Quarto → PDF)
├── references.bib                # Bibliography
├── _targets.R                    # targets pipeline configuration
├── R/                            # R scripts for data prep, fitting, and plotting
├── stan/                         # Stan model files
├── data/                         # Primate data and phylogenies
├── figures/                      # All figures (main text and SI)
└── _extensions/                  # Quarto extensions
```

## Getting started

### Prerequisites

- [R](https://www.r-project.org/) (4.4.3)
- [CmdStan](https://mc-stan.org/cmdstanr/) (2.37.0)
- [Quarto](https://quarto.org/) (>= 1.6)

### Installing dependencies

All R package dependencies (including exact versions and GitHub commit SHAs) are managed with [`renv`](https://rstudio.github.io/renv/). After cloning the repo:

```r
# install.packages("renv")
renv::restore()
```

Then install CmdStan if you don't already have it:

```r
cmdstanr::install_cmdstan(version = "2.37.0")
```

### Running the analysis pipeline

The analysis is orchestrated using the [`targets`](https://docs.ropensci.org/targets/) R package. All targets — including the synthetic example, simulation-based calibration, primate GDPM analysis, and manuscript rendering — are run together:

```bash
./run.sh            # full pipeline (several hours)
./run.sh --quick    # smoke test with reduced simulations/iterations (~10-15 min)
```

Or equivalently from R:

```r
library(targets)
tar_make()  # full pipeline

# Quick mode:
# Sys.setenv(GDPM_QUICK = "TRUE"); tar_make()
```

The quick mode (`GDPM_QUICK=TRUE`) reduces SBC simulations from 500 to 10 per configuration, and runs all MCMC fits with fewer chains and iterations. This is useful for verifying the pipeline runs end-to-end before committing to a full run.

Individual targets can be loaded with `tar_load()` or `tar_read()`.

## Help

For questions about the paper or code, please email erikjringen@gmail.com.

For questions about the `coevolve` package, see https://github.com/ScottClaessens/coevolve/issues.

## Authors

- Erik J. Ringen (erikjringen@gmail.com)
- Scott Claessens
- Jordan S. Martin
- Adrian V. Jaeggi

## License

See [LICENSE.md](LICENSE.md).
