#%%
import arviz as az
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
import enum
import collections
import pandas as pd

from typing import List
from typing import Dict
from typing import Union
from typing import NamedTuple
tfb = tfp.bijectors

USE_XLA = False
NUMBER_OF_CHAINS = 2
NUMBER_OF_BURNIN = 500
NUMBER_OF_SAMPLES = 500
NUMBER_OF_LEAPFROG_STEPS = 4

def dataframe_to_tensors(
    name: str,
    df: pd.DataFrame,
    columns: Union[List, Dict],
    default_type=tf.float32,
) -> NamedTuple:
    """name : Name of the dataset
    df : pandas dataframe
    colums : a list of names that have the same type or
             a dictionary where keys are the column names and values are the tensorflow type (e.g. tf.float32)
    """
    if isinstance(columns, dict):
        column_names = columns.keys()
        fields = [tf.cast(df[k].values, dtype=v) for k, v in columns.items()]
    else:
        column_names = columns
        fields = [tf.cast(df[k].values, dtype=default_type) for k in column_names]

    # build the cls
    tuple_cls = collections.namedtuple(name, column_names)
    # build the obj
    return tuple_cls._make(fields)

def _trace_to_arviz(
    trace=None,
    sample_stats=None,
    observed_data=None,
    prior_predictive=None,
    posterior_predictive=None,
    inplace=True,
):

    if trace is not None and isinstance(trace, dict):
        trace = {k: v.numpy() for k, v in trace.items()}
    if sample_stats is not None and isinstance(sample_stats, dict):
        sample_stats = {k: v.numpy().T for k, v in sample_stats.items()}
    if prior_predictive is not None and isinstance(prior_predictive, dict):
        prior_predictive = {k: v[np.newaxis] for k, v in prior_predictive.items()}
    if posterior_predictive is not None and isinstance(posterior_predictive, dict):
        if isinstance(trace, az.InferenceData) and inplace == True:
            return trace + az.from_dict(posterior_predictive=posterior_predictive)
        else:
            trace = None

    return az.from_dict(
        posterior=trace,
        sample_stats=sample_stats,
        prior_predictive=prior_predictive,
        posterior_predictive=posterior_predictive,
        observed_data=observed_data,
    )


@tf.function(autograph=False, experimental_compile=USE_XLA)
def run_hmc_chain(
    init_state,
    bijectors,
    step_size,
    target_log_prob_fn,
    num_leapfrog_steps=NUMBER_OF_LEAPFROG_STEPS,
    num_samples=NUMBER_OF_SAMPLES,
    burnin=NUMBER_OF_BURNIN,
):
    def _trace_fn_transitioned(_, pkr):
        return pkr.inner_results.inner_results.log_accept_ratio

    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn, num_leapfrog_steps=num_leapfrog_steps, step_size=step_size
    )

    inner_kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=hmc_kernel, bijector=bijectors
    )

    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel,
        target_accept_prob=0.8,
        num_adaptation_steps=int(0.8 * burnin),
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )

    results, sampler_stat = tfp.mcmc.sample_chain(
        num_results=num_samples,
        num_burnin_steps=burnin,
        current_state=init_state,
        kernel=kernel,
        trace_fn=_trace_fn_transitioned,
    )

    return results, sampler_stat

def sample_posterior(
    jdc,
    observed_data,
    params,
    init_state=None,
    bijectors=None,
    step_size=0.1,
    num_chains=NUMBER_OF_CHAINS,
    num_samples=NUMBER_OF_SAMPLES,
    burnin=NUMBER_OF_BURNIN,
):

    if init_state is None:
        init_state = list(jdc.sample(num_chains)[:-1])

    if bijectors is None:
        bijectors = [tfb.Identity() for i in init_state]

    target_log_prob_fn = lambda *x: jdc.log_prob(x + observed_data)

    results, sample_stats = run_hmc_chain(
        init_state,
        bijectors,
        step_size=step_size,
        target_log_prob_fn=target_log_prob_fn,
        num_samples=num_samples,
        burnin=burnin,
    )

    stat_names = ["mean_tree_accept"]
    sampler_stats = dict(zip(stat_names, [sample_stats]))

    transposed_results = []

    for r in results:
        if len(r.shape) == 2:
            transposed_shape = [1, 0]
        elif len(r.shape) == 3:
            transposed_shape = [1, 0, 2]
        else:
            transposed_shape = [1, 0, 2, 3]

        transposed_results.append(tf.transpose(r, transposed_shape))

    posterior = dict(zip(params, transposed_results))

    az_trace = _trace_to_arviz(trace=posterior, sample_stats=sampler_stats)

    return posterior, az_trace


#%%
tfd = tfp.distributions
tfb = tfp.bijectors
Root = tfd.JointDistributionCoroutine.Root
#%%
import pandas as pd
def model_binomial(applications):
    def _generator():
        a = yield Root(tfd.Sample(tfd.Normal(loc=0.0, scale=1.5), sample_shape=1))
        logit = a[..., tf.newaxis]

        T = yield tfd.Independent(
            tfd.Binomial(total_count=applications, logits=logit),
            reinterpreted_batch_ndims=1,
        )

    return tfd.JointDistributionCoroutine(_generator, validate_args=False)
d = pd.read_csv('/home/sosa/BI/data/UCBadmit.csv', sep = ';')
tdf = dataframe_to_tensors(
    "UCBAdmit",
    d,
    {"applications": tf.float32, "admit": tf.float32, "reject": tf.float32},
)

#%%
jdc_model_binom = model_binomial(tdf.applications)

init_state = [tf.zeros([NUMBER_OF_CHAINS])]

bijectors = [tfb.Identity()]


#%%
observed_data = (tdf.admit, tdf.reject)

posterior_model_binom, trace_model_binom = sample_posterior(
    jdc_model_binom,
    observed_data=observed_data,
    init_state=init_state,
    bijectors=bijectors,
    params=["alpha"],
)
