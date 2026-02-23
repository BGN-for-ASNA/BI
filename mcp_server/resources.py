"""
Enhanced MCP Resources with Quarto Documentation Integration.

This module loads your existing Quarto documentation files and exposes them
as MCP resources, allowing AI assistants to read and learn from your examples.
"""

from typing import List, Dict
import json
from pathlib import Path


# Dataset descriptions (existing)
DATASETS = {
    "howell1": {
        "name": "Howell1",
        "description": "Demographic data from the Dobe area !Kung San people. Contains height, weight, age, and sex for 544 individuals.",
        "features": ["height", "weight", "age", "male"],
        "samples": 544
    },
    "milk": {
        "name": "Primate Milk Composition",
        "description": "Milk composition data for various primate species including calories, fat, protein, lactose percentages.",
        "features": ["clade", "species", "kcal.per.g", "perc.fat", "perc.protein", "perc.lactose", "mass", "neocortex.perc"],
        "samples": 29
    },
    "iris": {
        "name": "Iris Dataset",
        "description": "Classic iris dataset with sepal and petal measurements for three species.",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        "samples": 150
    },
    "chimpanzees": {
        "name": "Chimpanzee Behavioral Data",
        "description": "Data from experiments on chimpanzee behavior and social learning.",
        "features": ["Various behavioral and experimental outcome measures"],
        "samples": "Multiple observations"
    },
    "reedfrogs": {
        "name": "Reed Frogs",
        "description": "Experimental data on reed frog tadpole survival under various conditions.",
        "features": ["Survival rates under different predation and density conditions"],
        "samples": 48
    },
    "tulips": {
        "name": "Tulips Growth Experiment",
        "description": "Experimental data on tulip growth under different water and shade treatments.",
        "features": ["blooms", "water", "shade"],
        "samples": 27
    },
    "ucbadmit": {
        "name": "UC Berkeley Admissions",
        "description": "Famous dataset illustrating Simpson's paradox. Graduate admissions data by department and gender.",
        "features": ["dept", "applicant.gender", "admit", "reject"],
        "samples": 12
    },
    "trolley": {
        "name": "Trolley Problem",
        "description": "Moral judgment experiment data using trolley problem scenarios.",
        "features": ["Various experimental conditions and moral judgment responses"],
        "samples": "Large dataset"
    },
    "elephants": {
        "name": "Elephants",
        "description": "Data from experiments on elephant matriarch age and survival.",
        "features": ["age", "matriarch"],
        "samples": 10
    },
    "waffle_divorce": {
        "name": "Waffle House and Divorce",
        "description": "State-level data on divorce rates, marriage rates, and Waffle House locations.",
        "features": ["Location", "Divorce", "Marriage", "MedianAgeMarriage", "WaffleHouses"],
        "samples": 50
    }
}


# Path to Documentation folder
DOCS_PATH = Path(__file__).parent.parent / "Documentation"


STAN_EXAMPLES_PATH = Path(__file__).parent / "data" / "stan_bi_examples.json"
STAN_SEMANTICS_PATH = Path(__file__).parent / "data" / "stan_semantics.yaml"


# Mapping of documentation resources to their Quarto files
QUARTO_DOCS = {
    # Getting Started
    "getting_started": "get started.qmd",
    "introduction": "0.Introduction.qmd",
    
    # Basic Regression Models
    "linear_regression": "1. Linear Regression for continuous variable.qmd",
    "multiple_regression": "2. Multiple continuous Variables.qmd",
    "interactions": "3. Interaction between continuous variables.qmd",
    "categorical_predictors": "4. Categorical variable.qmd",
    
    # GLM Models
    "binomial_model": "5. Binomial model.qmd",
    "beta_binomial": "6. Beta binomial model.qmd",
    "poisson_model": "7. Poisson model.qmd",
    "gamma_poisson": "8. Gamma-Poisson.qmd",
    "poisson_offset": "8. Poisson mode with offset.qmd",
    "categorical_outcomes": "9. Categorical model.qmd",
    "dirichlet_model": "10. Dirichlet model.qmd",
    "multinomial_model": "10. Multinomial model.qmd",
    "zero_inflated": "11. Zero inflated.qmd",
    
    # Advanced Models
    "survival_analysis": "12. Survival analysis.qmd",
    "varying_intercepts": "13. Varying intercepts.qmd",
    "varying_slopes": "14. Varying slopes.qmd",
    "gaussian_processes": "15. Gaussian processes.qmd",
    "measurement_error": "16. Measuring error.qmd",
    "missing_data": "17. Missing data.qmd",
    
    # Machine Learning
    "pca": "19. PCA.qmd",
    "gmm": "20. GMM.qmd",
    "dpmm": "21. DPMM.qmd",
    "bnn": "27. BNN.qmd",
    
    # Network Models
    "network_model": "22. Network model.qmd",
    "network_block_model": "23. Network with block model.qmd",
    "network_biases": "24. Network control for data collection biases.qmd",
    "network_metrics": "25. Network Metrics.qmd",
    "nbda": "26. Network Based Diffusion analysis (wip).qmd",
    
    # API Reference
    "api_distributions": "api_dist.qmd",
    "api_diagnostics": "api_diag.qmd",
    "api_manipulation": "api_manip.qmd",
    "mcp_usage_guide": "mcp_usage_guide.md",
}


# Fallback documentation (for when Quarto files aren't available)
FALLBACK_DOCS = {
    "getting_started": """# BayesInference - Getting Started

BayesInference (BI) provides a unified probabilistic programming library for Bayesian inference.

## Quick Start

```python
from BI import bi

# Initialize BI
m = bi(platform="cpu")  # Options: "cpu", "gpu", "tpu"

# Generate some data
x = m.dist.normal(0, 1, shape=(100,), sample=True)
y = m.dist.normal(0.2 + 0.6 * x, 1.2, sample=True)

# Define a Bayesian linear regression model
def linear_model(x, y):
    alpha = m.dist.normal(loc=0, scale=1, name="alpha")
    beta = m.dist.normal(loc=0, scale=1, name="beta")
    sigma = m.dist.exponential(1, name="sigma")
    mu = alpha + beta * x
    m.dist.normal(mu, sigma, obs=y)

# Fit the model
m.fit(linear_model, num_warmup=1000, num_samples=1000, num_chains=1)

# Display results
m.summary()

# Plot results
m.plot_trace()
```

## Key Features

- **Unified API** across Python, Julia, and R
- **JAX-powered** for CPU/GPU/TPU acceleration
- **Rich diagnostics** via ArviZ integration
- **Extensive distribution library** from NumPyro
""",
    "distributions": """# Available Distributions in BayesInference

All distributions are accessed via `m.dist.<distribution_name>`.

## Continuous Distributions

- `normal(loc, scale, ...)` - Normal/Gaussian distribution
- `uniform(low, high, ...)` - Uniform distribution
- `student_t(df, loc, scale, ...)` - Student's t-distribution
- `cauchy(loc, scale, ...)` - Cauchy distribution
- `halfcauchy(scale, ...)` - Half-Cauchy distribution
- `halfnormal(scale, ...)` - Half-Normal distribution
- `gamma(concentration, rate, ...)` - Gamma distribution
- `inverse_gamma(concentration, rate, ...)` - Inverse Gamma
- `exponential(rate, ...)` - Exponential distribution
- `beta(concentration0, concentration1, ...)` - Beta distribution
- `laplace(loc, scale, ...)` - Laplace distribution
- `log_normal(loc, scale, ...)` - Log-Normal distribution
- `weibull(concentration, scale, ...)` - Weibull distribution

## Discrete Distributions

- `bernoulli(probs, ...)` - Bernoulli distribution
- `binomial(total_count, probs, ...)` - Binomial distribution
- `poisson(rate, ...)` - Poisson distribution
- `negative_binomial(total_count, probs, ...)` - Negative Binomial
- `categorical(probs, ...)` - Categorical distribution
- `zero_inflated_poisson(gate, rate, ...)` - Zero-Inflated Poisson

## Multivariate Distributions

- `multivariate_normal(loc, covariance_matrix, ...)` - Multivariate Normal
- `dirichlet(concentration, ...)` - Dirichlet distribution
- `multinomial(total_count, probs, ...)` - Multinomial distribution
- `lkj(dimension, concentration, ...)` - LKJ correlation distribution
- `wishart(df, scale_matrix, ...)` - Wishart distribution

## Common Parameters

- `name` - Variable name for the parameter (required for latent variables)
- `obs` - Observed data (for likelihood)
- `shape` - Shape of the distribution
- `sample` - Set to `True` for data simulation/prior predictive sampling

## Example Usage

```python
# Prior distributions
alpha = m.dist.normal(0, 10, name="alpha")
beta = m.dist.normal(0, 1, name="beta", shape=(3,))
sigma = m.dist.exponential(1, name="sigma")

# Likelihood
mu = alpha + beta @ X
m.dist.normal(mu, sigma, obs=y)
```
""",
    "mcp_usage_guide": """# BayesInference MCP Usage Guide

This guide explains how to use the BayesInference (BI) package correctly through this MCP server.

## Core Concepts

1.  **The `m` Object**: In all tools that accept Python code (like `fit_model` or `simulate_data`), a pre-initialized BayesInference instance named `m` is available. **Always use `m.dist.*` for distributions.**
2.  **Importing Data**: 
    - **Tabular Data**: Use `m.data(path, sep=';')` to load CSV files.
    - **Non-Tabular Data**: Assign a dictionary directly to `m.data_on_model = dict(ID1=Value1, ID2=Value2)`.
    - **Fitting**: Once data is loaded, you can call `m.fit(model)` without passing data explicitly.
3.  **Data Simulation**: Use the `simulate_data` tool. Inside the code, use `m.dist.<dist_name>(..., sample=True)` to generate data.
4.  **Model Definition**: Use the `fit_model` tool. Your code must define a function named `model` that uses `m.dist.*` for priors and likelihood.
5.  **Diagnostics**: Use `get_summary` and `get_diagnostics` after fitting a model.

## Example: Zero-Inflated Poisson with Varying Intercepts

### 1. Simulate Data
Use `simulate_data` with:
```python
import jax.numpy as jnp
N = 200
num_groups = 10
group_idx = m.randint(0, num_groups, (N,))
mu_alpha = 1.5
sigma_alpha = 0.5
alpha = m.dist.normal(mu_alpha, sigma_alpha, shape=(num_groups,), sample=True)
zi_prob = 0.2

lambda_ = jnp.exp(alpha[group_idx])
y = m.dist.zero_inflated_poisson(gate=zi_prob, rate=lambda_, sample=True)
```

### 2. Fit Model
Use `fit_model` with:
```python
def model(y, group_idx, num_groups):
    # Priors
    mu_alpha = m.dist.normal(0.0, 10.0, name='mu_alpha')
    sigma_alpha = m.dist.half_normal(10.0, name='sigma_alpha')
    
    with m.dist.plate('groups', num_groups):
        alpha = m.dist.normal(mu_alpha, sigma_alpha, name='alpha')
    
    # Zero-inflation probability
    zi_prob = m.dist.beta(1.0, 1.0, name='zi_prob')
    
    # Linear predictor
    lambda_ = jnp.exp(alpha[group_idx])
    
    # Likelihood
    m.dist.zero_inflated_poisson(gate=zi_prob, rate=lambda_, obs=y)
```

### 3. Diagnostics
Call `get_summary(model_id='default')`.
"""
}


def load_quarto_file(filename: str) -> str:
    """
    Load a Quarto documentation file.
    
    Args:
        filename: Name of the Quarto file
        
    Returns:
        Content of the file or error message
    """
    try:
        filepath = DOCS_PATH / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"Documentation file not found: {filename}"
    except Exception as e:
        return f"Error loading documentation: {str(e)}"


def get_dataset_resource(dataset_name: str) -> str:
    """
    Get dataset information as a resource.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        JSON string with dataset information
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return json.dumps(DATASETS[dataset_name], indent=2)


def get_docs_resource(doc_name: str) -> str:
    """
    Get documentation as a resource.
    Tries to load from Quarto files first, falls back to hardcoded docs.
    
    Args:
        doc_name: Name of the documentation
        
    Returns:
        Documentation string
    """
    # Try to load from Quarto file
    if doc_name in QUARTO_DOCS:
        content = load_quarto_file(QUARTO_DOCS[doc_name])
        if not content.startswith("Documentation file not found") and \
           not content.startswith("Error loading documentation"):
            return content
    
    # Fall back to hardcoded documentation
    if doc_name in FALLBACK_DOCS:
        return FALLBACK_DOCS[doc_name]
    
    raise ValueError(f"Unknown documentation: {doc_name}")


def list_available_datasets() -> List[str]:
    """Get list of available dataset names."""
    return list(DATASETS.keys())


def list_available_docs() -> List[str]:
    """Get list of available documentation names (includes all Quarto docs)."""
    # Combine Quarto docs and fallback docs
    all_docs = set(list(QUARTO_DOCS.keys()) + list(FALLBACK_DOCS.keys()))
    return sorted(list(all_docs))


def get_docs_by_category() -> Dict[str, List[str]]:
    """
    Get documentation organized by category.
    
    Returns:
        Dictionary mapping categories to lists of doc names
    """
    return {
        "Getting Started": [
            "getting_started",
            "introduction",
            "mcp_usage_guide"
        ],
        "Basic Regression": [
            "linear_regression",
            "multiple_regression",
            "interactions",
            "categorical_predictors"
        ],
        "Generalized Linear Models": [
            "binomial_model",
            "beta_binomial",
            "poisson_model",
            "gamma_poisson",
            "poisson_offset",
            "categorical_outcomes",
            "dirichlet_model",
            "multinomial_model",
            "zero_inflated"
        ],
        "Advanced Models": [
            "survival_analysis",
            "varying_intercepts",
            "varying_slopes",
            "gaussian_processes",
            "measurement_error",
            "missing_data"
        ],
        "Machine Learning": [
            "pca",
            "gmm",
            "dpmm",
            "bnn"
        ],
        "Network Models": [
            "network_model",
            "network_block_model",
            "network_biases",
            "network_metrics",
            "nbda"
        ],
        "API Reference": [
            "api_distributions",
            "api_diagnostics",
            "api_manipulation"
        ]
    }


def get_stan_conversion_examples() -> str:
    """
    Get the list of all paired Stan to BI conversion examples.
    """
    try:
        if STAN_EXAMPLES_PATH.exists():
            with open(STAN_EXAMPLES_PATH, 'r', encoding='utf-8') as f:
                return f.read()
        return json.dumps({"error": "Stan examples file not found."})
    except Exception as e:
        return json.dumps({"error": f"Error loading Stan examples: {str(e)}"})


def get_stan_conversion_example(example_id: str) -> str:
    """
    Get a specific Stan to BI conversion example by ID.
    """
    try:
        if STAN_EXAMPLES_PATH.exists():
            with open(STAN_EXAMPLES_PATH, 'r', encoding='utf-8') as f:
                examples = json.load(f)
                for ex in examples:
                    if ex.get("id") == example_id:
                        return json.dumps(ex, indent=2)
                return json.dumps({"error": f"Example ID '{example_id}' not found."})
        return json.dumps({"error": "Stan examples file not found."})
    except Exception as e:
        return json.dumps({"error": f"Error loading Stan example: {str(e)}"})


def get_stan_semantics() -> str:
    """
    Get the formal Stan -> BI semantic mapping specification.
    """
    try:
        if STAN_SEMANTICS_PATH.exists():
            with open(STAN_SEMANTICS_PATH, 'r', encoding='utf-8') as f:
                return f.read()
        return "Stan semantics file not found."
    except Exception as e:
        return f"Error loading Stan semantics: {str(e)}"

