# BI
Bayesian Inference using TensorFlow Probability

# Why?
## 1.  Easy Model Building:

The following linear regression model (rethinking 4.Geocentric Models): 
```math
height∼Normal(μ,σ)
```
```math
μ=α+β(weight)
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
## 2.  Fast Computation through GPU
If a GPU is detected, the model computation can be run on it.

## 3.  Offer a graphical user interface.
![image](https://github.com/BGN-for-ASNA/BI/assets/22368172/5ce6dd41-1188-4cfe-83f1-481ce0992787)
