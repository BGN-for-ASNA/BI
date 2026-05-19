import os
import sys
import json
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel

def main():
    # Load Data
    with open('stan_data.json', 'r') as f:
        data = json.load(f)
    
    # Run Stan
    print("Running Stan model for reference samples...")
    stan_file = 'cg_vocal_repertoires/model0.stan'
    model_stan = CmdStanModel(stan_file=stan_file)
    fit_stan = model_stan.sample(data='stan_data.json', seed=42, iter_sampling=1000, iter_warmup=1000, chains=2)
    
    # Save samples
    samples = fit_stan.draws_pd()
    samples.to_csv('stan_reference_samples.csv', index=False)
    print("Samples saved to stan_reference_samples.csv")

if __name__ == "__main__":
    main()
