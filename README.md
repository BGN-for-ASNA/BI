# BI
Bayesian Inference using TensorFlow Probability
Currently, the package provides:

+ Data manipulation:
    + One-hot encoding
    + Conversion of index variables
    + Scaling
      
+ Models:
    + Single models
    + Multiple models in one
    + Continuous variables and index variables
    + Poisson, Binomial, Normal, zero-inflated, negative binomial, Multinomial disguised as Binomial or Poisson, Beta-binomial
    + Varying intercepts and effects
    + Gaussian processes

+ Model diagnostics (using ARVIZ):
    + Data frame with summary statistics
    + Plot posterior densities
    + Bar plot of the autocorrelation function (ACF) for a sequence of data
    + Plot rank order statistics of chains
    + Forest plot to compare HDI intervals from a number of distributions
    + Compute the widely applicable information criterion
    + Compare models based on their expected log pointwise predictive density (ELPD)
    + Compute estimate of rank normalized split-R-hat for a set of traces
    + Calculate estimate of the effective sample size (ESS)
    + Pair plot
    + Density plot
    + ESS evolution plot
      
# Model and Results Comparisons
This package has been built following the Rethinking Classes of 2024. Each week, new approaches have been implemented and validated with the main example of the corresponding week. All models can be found in the following [Jupyter notebook](https://github.com/BGN-for-ASNA/BI/blob/main/rethinking.ipynb). 

# Why?
## 1.  To learn

## 2.  Easy Model Building:
The following linear regression model (rethinking 4.Geocentric Models): 
```math
height∼Normal(μ,σ)
```
```math
μ=α+β*weight
```
```math 
α∼Normal(178,20)
```
```math
β∼Normal(0,10)
```
```math
σ∼Uniform(0,50)
```
    
can be declared in the package as
```
# Setup device------------------------------------------------
from main import*
m = bi(platform='cpu')


# Import data ------------------------------------------------
m.data('../data/Howell1.csv', sep=';') 
m.df = m.df[m.df.age > 18]
m.scale(['weight'])
# TODO: use jax arrays with hugging face package
m.data_to_model(['weight', 'height'])


 # define model ------------------------------------------------
def model(height, weight):
    s = dist.uniform( 0, 50, name = 's',shape = [1])
    a = dist.normal( 178, 20, name = 'a',shape= [1])
    b = dist.normal(  0, 1, name = 'b',shape= [1])   
    lk("y", Normal(a + b * weight , s), obs=height)

# Run sampler ------------------------------------------------
m.run(model) 
m.sampler.print_summary(0.89)
```            

# Todo 
1. GUI 
2. Helper functions
3. Documentation
4. Multinomial models to be run with the Multinomial distribution
5. Multiple likelihoods can have different types: independent models -> independent HMC, dependent priors -> 
6. Posterior needs to handle multiple likelihoods
7. Implementation of additional MCMC sampling methods
8. Float precision handling

