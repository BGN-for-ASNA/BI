
#%%
from BayesForge import bf
import numpy as np
import jax.numpy as jnp

# Setup device ------------------------------------------------
m = bf(platform='cpu')

# Import data & manipulation ---------------------------------
data_path = m.load.mastectomy(only_path=True)
m.data(data_path, sep=',')
#%%
# 'yes'/'no' -> 1/0
m.df.metastasized = (m.df.metastasized.values == "yes").astype(np.int64)

# Build survival (discrete-time) object ----------------------
# interval_length=3 matches the PyMC reference notebook.
m.models.survival.import_time_even(m.df.time.values, m.df.event.values, interval_length=3)
m.models.survival.import_covF(m.df.metastasized.values, ["metastasized"])

# Plot censoring --------------------------------------------
m.models.survival.plot_censoring(cov='metastasized')
#%%
# Priors (overridable before fit) --------------------------
# Baseline_rate ~ Gamma(0.01, 0.01) per interval
# Hazard_rate_metastasized ~ Normal(0, 10)
m.models.survival.baseline_rate_prior = (0.01, 0.01)
m.models.survival.hazard_rate_prior_scale = 10.0

# Run MCMC ------------------------------------------------
# Model is built in: m.models.survival.model (no hand-written fn).
m.fit(m.models.survival.model, num_samples=1000, num_warmup=1000,
      num_chains=2, progress_bar=False, seed=42)

# Summary ------------------------------------------------
print(m.summary())

# Plot hazards and survival function -----------------------
m.models.survival.plot_surv(beta='Hazard_rate_metastasized')
