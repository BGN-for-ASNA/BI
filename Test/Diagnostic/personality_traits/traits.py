import jax.numpy as jnp

def trait_model(m, y, individual):
    """
    Model for personality traits and repeated measures using variance partitioning.
    
    Args:
        m: BayesForge (BF) model instance
        y: The behavioral/trait measure (array of observations)
        individual: The integer ID for each subject (array matching len(y))
    """
    # 1. Between-individual variation (tau)
    tau = m.dist.halfnormal(1.0, name='tau_between')
    
    # 2. Extract latent traits for each individual (non-centered parameterization)
    n_individuals = len(jnp.unique(individual))
    z_id = m.dist.normal(0, 1, shape=(n_individuals,), name='z_id')
    alpha_id = m.deterministic('alpha_id', z_id * tau)
    
    # 3. Overall average behavior
    alpha = m.dist.normal(0, 5, name='alpha')
    
    # 4. Within-individual variation (sigma)
    sigma = m.dist.halfnormal(1.0, name='sigma_within')
    
    # Likelihood
    mu = alpha + alpha_id[individual]
    m.dist.normal(mu, sigma, obs=y)


def calculate_repeatability(m):
    """
    Calculates Repeatability, also known as the Intraclass Correlation Coefficient (ICC).
    Repeatability measures the proportion of total variance explained by individual differences:
    R = tau^2 / (tau^2 + sigma^2)
    
    Args:
        m: BayesForge (BF) model instance (after inference is complete)
        
    Returns:
        R_samples: The Repeatability calculated for every MCMC draw
        stats: Dictionary containing the mean and 95% HDI
    """
    # Assuming posteriors have been fetched correctly
    tau_samples = m.posteriors['tau_between']
    sigma_samples = m.posteriors['sigma_within']
    
    # Calculate ICC for every MCMC draw
    R_samples = tau_samples**2 / (tau_samples**2 + sigma_samples**2)
    
    # Summarize the Repeatability metric
    mean_R = R_samples.mean()
    hdi_lower = jnp.percentile(R_samples, 2.5)
    hdi_upper = jnp.percentile(R_samples, 97.5)
    
    stats = {
        'mean': mean_R,
        'hdi_2.5': hdi_lower,
        'hdi_97.5': hdi_upper
    }
    
    return R_samples, stats


def extract_latent_traits(m):
    """
    Extracts the latent trait expressions (unobservable personality scores) of each individual.
    
    Args:
        m: BayesForge (BF) model instance (after inference is complete)
        
    Returns:
        traits_summary: Dictionary containing posterior means and 95% HDIs for each individual's alpha_id
    """
    alpha_id_samples = m.posteriors['alpha_id']
    
    mean_traits = alpha_id_samples.mean(axis=0)
    hdi_lower = jnp.percentile(alpha_id_samples, 2.5, axis=0)
    hdi_upper = jnp.percentile(alpha_id_samples, 97.5, axis=0)
    
    traits_summary = {
        'mean': mean_traits,
        'hdi_2.5': hdi_lower,
        'hdi_97.5': hdi_upper
    }
    
    return traits_summary
