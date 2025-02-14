# BI
Bayesian Inference using numpyro.

Currently, the package provides:

+ Data manipulation:
    + One-hot encoding
    + Conversion of index variables
    + Scaling
      
+ Models (Using Numpyro):
  
    + [Linear Regression for continuous variable](Documentation/1.&#32;Linear&#32;Regression&#32;for&#32;continuous&#32;variable.qmd)
    + [Multiple continuous Variable](Documentation/2.&#32;Multiple&#32;continuous&#32;Variables.qmd)
    + [Interaction between variables](Documentation/3.&#32;Interaction&#32;between&#32;continuous&#32;variables.qmd)
    + [Categorical variable](Documentation/4.&#32;Categorical&#32;variable.qmd)
    + [Binomial model](Documentation/5.&#32;Binomial&#32;model.qmd)
    + [Beta binomial](Documentation/6.&#32;Beta&#32;binomial&#32;model.qmd)
    + [Poisson model](Documentation/7.&#32;Poisson&#32;model.qmd)
    + [Gamma-Poisson](Documentation/8.&#32;Gamma-Poisson.qmd)
    + [Multinomial](Documentation/9.&#32;Multinomial&#32;model.qmd)    
    + [Dirichlet model](Documentation/10.&#32;Dirichlet&#32;model&#32;(wip).qmd)
    + [Zero inflated](Documentation/11.&#32;Zero&#32;inflated.qmd)
    + [Varying intercept](Documentation/12.&#32;Varying&#32;intercepts.qmd)
    + [Varying slopes](Documentation/13.&#32Varying&#32slopes.qmd)
    + [Gaussian processes](Documentation/14.&#32;Gaussian&#32;processes&#32;(wip).qmd)  
    + [Measuring error](Documentation/15.&#32;Measuring&#32;error&#32;(wip).qmd) 
    + [Latent variable](Documentation/17.&#32;Latent&#32;variable&#32;(wip).qmd) 
    + [PCA](Documentation/18.&#32;PCA&#32;(wip).qmd) 
    + [Network model](Documentation/18.&#32;Network&#32;model.qmd) 
    + [Network with block model](Documentation/19.&#32;Network&#32;with&#32;block&#32;model.qmd)
    + [Network control for data collection biases ](Documentation/20.&#32;Network&#32;control&#32;for&#32;data&#32;collection&#32;biases&#32;(wip).qmd)

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
m.data_to_model(['weight', 'height'])


 # Define model ------------------------------------------------
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

