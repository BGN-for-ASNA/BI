# BI
Bayesian Inference using TensorFlow Probability
Currently, the package provides:

+ Data manipulation:
    + One-hot encoding
    + Conversion of index variables
    + Scaling
      
+ Models (Using Numpyro):
  
    + [Linear Regression for continuous variable](doc/1.&#32;Linear&#32;Regression&#32;for&#32;continuous&#32;variable.md)
    + [Multiple continuous Variable](doc/2.&#32;Multiple&#32;continuous&#32;Variables.md)
    + [Interaction between continuous variable](doc/3.&#32;Interaction&#32;between&#32;continuous&#32;variables.md)
    + [Categorical variable](doc/4.&#32;Categorical&#32;variable.md)
    + [Binomial regression](doc/5.&#32;Binomial&#32;regression.md)
    + [Poisson regression](doc/6.&#32;Poisson&#32;regression.md)
    + [Negative binomial](doc/7.&#32;Negative&#32;binomial.md)
    + [Multinomial](doc/8.&#32;Multinomial.md)
    + [Beta binomial](doc/9.&#32;Beta&#32;binomial.md)
    + [Negative-binomial](doc/10.&#32;Negative-binomial.md)
    + [Zero inflated](doc/11.&#32;Zero&#32;inflated.md)
    + [Varying interceps](doc/12.&#32;Varying&#32;interceps.md)
    + [Varying slopes](doc/13.&#32;Varying&#32;slopes.md)
    + [Multiple varying slopes](doc/14.Multiple&#32;varying&#32;slopes.md) (WIP)
    + [Continuous varying slope](doc/15.&#32;Continuous&#32;varying&#32;slopes.md) (WIP)
    + [Gaussian processes](doc/16.&#32;Gaussian&#32;processes.md) (WIP)    
    + [Measuring error](doc/17.&#32;Measuring&#32;error.md) (WIP)
    + [Missing data](doc/18.&#32;Missing&#32;data.md) (WIP)
    + [Modeling Network](doc/19.&#32;Modeling&#32;Network.md)

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

