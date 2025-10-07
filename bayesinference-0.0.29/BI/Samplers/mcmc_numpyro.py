import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import condition
from BI.Data.manip import manip
import jax 

def mcmc_numpyro(
    model = None, 
    potential_fn=None,
    kinetic_fn=None,
    step_size=1.0,
    inverse_mass_matrix=None,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=False,
    target_accept_prob=0.8,
    trajectory_length=None,
    max_tree_depth=10,
    init_strategy= numpyro.infer.init_to_uniform,
    find_heuristic_step_size=False,
    forward_mode_differentiation=False,
    regularize_mass_matrix=True,
    
    num_warmup = 500,
    num_samples = 500,
    num_chains=1,
    thinning=1,
    postprocess_fn=None,
    chain_method="parallel",
    progress_bar=True,
    jit_model_args=False,
    seed = 0
):
    return MCMC(
                NUTS(
                    model,
                    potential_fn=potential_fn,
                    kinetic_fn=kinetic_fn,
                    step_size=step_size,
                    inverse_mass_matrix=inverse_mass_matrix,
                    adapt_step_size=adapt_step_size,
                    adapt_mass_matrix=adapt_mass_matrix,
                    dense_mass=dense_mass,
                    target_accept_prob=target_accept_prob,
                    trajectory_length=trajectory_length,
                    max_tree_depth=max_tree_depth,
                    init_strategy=init_strategy,
                    find_heuristic_step_size=find_heuristic_step_size,
                    forward_mode_differentiation=forward_mode_differentiation,
                    regularize_mass_matrix=regularize_mass_matrix
                ), 
                num_warmup = num_warmup,
                num_samples = num_samples,
                num_chains=num_chains,
                thinning=thinning,
                postprocess_fn=postprocess_fn,
                chain_method=chain_method,
                progress_bar=progress_bar,
                jit_model_args=jit_model_args
            )

 