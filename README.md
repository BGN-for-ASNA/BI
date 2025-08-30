# [Bayesian Inference](https://pypi.org/project/BayesInference/ "BI")

[BI](https://pypi.org/project/BayesInference/ "BI") software is available in both Python and R. It aims to unify the modeling experience by integrating an intuitive model-building syntax with the flexibility of low-level abstraction coding available but also pre-build function for high-level of abstraction and including hardware-accelerated computation for improved scalability.

Currently, the package provides:

-   Data manipulation:

    -   One-hot encoding
    -   Conversion of index variables
    -   Scaling

-   Models (Using [Numpyro](https://num.pyro.ai/en/latest/index.html#)):

    -   Linear Regression for continuous variable
    -   Multiple continuous Variable
    -   Interaction between variables
    -   Categorical variable
    -   Binomial model
    -   Beta binomial
    -   Poisson model
    -   Gamma-Poisson
    -   Multinomial
    -   Dirichlet model
    -   Zero inflated
    -   Varying intercept
    -   Varying slopes
    -   Gaussian processes
    -   Measuring error
    -   Latent variable\]
    -   PCA
    -   GMM
    -   DPMM
    -   Network model
    -   Network with block model
    -   Network control for data collection biases
    -   BNN

-   Model diagnostics (using [ARVIZ](https://python.arviz.org/en/stable/)):

    -   Data frame with summary statistics
    -   Plot posterior densities
    -   Bar plot of the autocorrelation function (ACF) for a sequence of data
    -   Plot rank order statistics of chains
    -   Forest plot to compare HDI intervals from a number of distributions
    -   Compute the widely applicable information criterion
    -   Compare models based on their expected log pointwise predictive density (ELPD)
    -   Compute estimate of rank normalized split-R-hat for a set of traces
    -   Calculate estimate of the effective sample size (ESS)
    -   Pair plot
    -   Density plot
    -   ESS evolution plot

# Why?

## 1. To learn

## 2. Easy Model Building:

The following linear regression model (rethinking 4.Geocentric Models): $$
\text{height} \sim \mathrm{Normal}(\mu,\sigma)
$$

$$
\mu = \alpha + \beta \cdot \text{weight}
$$

$$
\alpha \sim \mathrm{Normal}(178,20)
$$

$$
\beta \sim \mathrm{Normal}(0,10)
$$

$$
\sigma \sim \mathrm{Uniform}(0,50)
$$

can be declared in the package as

```         
from BI import bi

# Setup device------------------------------------------------
m = bi(platform='cpu')

# Import Data & Data Manipulation ------------------------------------------------
# Import
from importlib.resources import files
data_path = files('BI.resources.data') / 'Howell1.csv'
m.data(data_path, sep=';') 
m.df = m.df[m.df.age > 18] # Manipulate
m.scale(['weight']) # Scale

# Define model ------------------------------------------------
def model(weight, height):    
    a = m.dist.normal(178, 20, name = 'a') 
    b = m.dist.lognormal(0, 1, name = 'b') 
    s = m.dist.uniform(0, 50, name = 's') 
    m.normal(a + b * weight , s, obs = height) 

# Run mcmc ------------------------------------------------
m.fit(model)  # Optimize model parameters through MCMC sampling

# Summary ------------------------------------------------
m.summary() # Get posterior distributions
```

# Todo

1.  GUI
2.  Documentation
3.  Implementation of additional MCMC sampling methods