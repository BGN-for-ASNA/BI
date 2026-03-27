import BI as bi
import jax.numpy as jnp
import pandas as pd
import numpy as np
from bi_phylogeny import model

def run():
    # Load data
    data_simple = pd.read_csv("data_simple.txt", sep="\s+")
    
    # In a real run, A comes from R. For testing, we can use an identity or load a saved one.
    # But wait, I have the nexus file. I can't easily compute VCV in python without ape.
    # So I'll wait for the R script to finish and save the covariance matrix.
    pass

if __name__ == "__main__":
    # Wait, I'll modify brms_fit_simple.R to also save the covariance matrix A and L.
    pass
