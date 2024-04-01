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
# Import data
d = pd.read_csv('data/Howell1.csv', sep=';')
weight_centered = d.weight - d.weight.mean()
height = d.height.values

# Declare model
model = dict(main = 'height~Normal(m,sigma)',
            likelihood = 'm = alpha + beta*weight_centered',
            prior1 = 'alpha~Normal(178, 20)',
            prior2 = 'beta ~ Normal(0,10)',
            prior3 = 'sigma ~ Uniform(0,50)')

# Run Hamiltonian MonteCarlo
Work in progress
```            
## 3.  No compilation time, fast computation, easy GPU computation configuration for big models.
If a GPU is detected, the model computation can be run on it.

## 4.  Offer a graphical user interface.
### Import data
![image](https://github.com/BGN-for-ASNA/BI/assets/22368172/cc1d023c-2ef4-4822-89ab-f0db96729387)

### Declared model
![image](https://github.com/BGN-for-ASNA/BI/assets/22368172/5ce6dd41-1188-4cfe-83f1-481ce0992787)

### Run model
Work in progress

# Todo 
1. Redo GUI (Implementation of new approaches has led to substantial modifications and GUI incompatibility)
2. Helper functions
3. Documentation
4. Multinomial models to be run with the Multinomial distribution
5. Multiple likelihoods can have different types: independent models -> independent HMC, dependent priors -> 
6. Posterior needs to handle multiple likelihoods
7. Implementation of additional MCMC sampling methods
8. Float precision handling

