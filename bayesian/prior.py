#%%
import tensorflow as tf
import numpy as np
import arviz as az
def prior_check(joint_dist, NUM_PRIOR_SAMPLES, N, obs):
    """_summary_

    Args:
        joint_dist (tfd.JointDistribution): TFD joint distribution function
        NUM_PRIOR_SAMPLES (int): Number of samples to perform
        N (int): Number of samples to perform
        data (tensor): The output tensor obtained by joint_dist.sample()
    """
    *prior_samples, prior_predictive = joint_dist.sample(NUM_PRIOR_SAMPLES)
    prior_trace = az.from_dict(
        observed_data={"observations": obs},
        prior_predictive={"observations": prior_predictive[tf.newaxis, ...]},
        coords={"observation": np.arange(N)},
        dims={"observations": ["observation"]},
    )
    ax = az.plot_ppc(prior_trace, group="prior", num_pp_samples=500)
    return ax