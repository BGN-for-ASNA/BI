import BI as bi
import jax.numpy as jnp

def model(data):
    # Unpack data
    phen = data['phen']
    cofactor = data['cofactor']
    phylo_idx = data['phylo_idx']
    A_cholesky = data['A_cholesky']
    num_species = data['num_species']
    
    # Priors
    intercept = bi.dist.normal(0, 50, name="Intercept")
    beta_cofactor = bi.dist.normal(0, 10, name="b_cofactor")
    
    # sd_phylo and sigma
    sd_phylo = bi.dist.half_normal(20, name="sd_phylo")
    sigma = bi.dist.half_normal(20, name="sigma")
    
    # Non-centered parameterization for species effects
    z_phylo = bi.dist.normal(jnp.zeros(num_species), 1.0, name="z_phylo")
    u_phylo = (A_cholesky @ z_phylo) * sd_phylo
    
    # Mean
    mu = intercept + beta_cofactor * cofactor + u_phylo[phylo_idx]
    
    # Likelihood
    bi.dist.normal(mu, sigma, name="obs", event=phen)
