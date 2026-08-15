from Utils import *
from BayesForge import bf, BF
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from importlib.resources import files

model_name = "13.Gaussian processes"

print(f'Running BF for {model_name}')
# Setup device------------------------------------------------
m = bf(platform='cpu', backend='tfp')

# Import Data & Data Manipulation ------------------------------------------------
data_path = m.load.kline2(only_path=True)
m.data(data_path, sep=';')

data_path2 = files('BayesForge.Resources') / 'islandsDistMatrix.csv'
islandsDistMatrix = pd.read_csv(data_path2, index_col=0)

m.data_to_model(['total_tools', 'population'])
m.data_on_model["society"] = jnp.arange(0, 10)
m.data_on_model["Dmat"] = islandsDistMatrix.values.astype(jnp.float32)

def model(Dmat, population, society, total_tools):
    a = yield m.dist.exponential(1,   name='a')
    b = yield m.dist.exponential(1,   name='b')
    g = yield m.dist.exponential(1,   name='g')

    etasq = yield m.dist.exponential(2,   name='etasq')
    rhosq = yield m.dist.exponential(0.5, name='rhosq')

    # Non-centered GP: k = L @ z, z ~ N(0, I)
    n = Dmat.shape[0]
    SIGMA = etasq * jnp.exp(-rhosq * jnp.square(Dmat))
    SIGMA = SIGMA.at[jnp.diag_indices(n)].add(0.01)
    L_chol = jnp.linalg.cholesky(SIGMA)
    z = yield m.dist.multivariate_normal(
        jnp.zeros(n, dtype=jnp.float32), jnp.eye(n, dtype=jnp.float32), name='k_z'
    )
    k = L_chol @ z

    lambda_ = a * population**b / g * jnp.exp(k[society])
    yield m.dist.poisson(lambda_, obs=total_tools)

# Run sampler ------------------------------------------------
print("Fitting BF model...")
m.fit(model, num_samples=1000, num_warmup=2000)
print("BF Summary:")
print(m.summary())

# 3. Stan Model
stan_code = """
functions{
  matrix cov_GPL2(matrix x, real sq_alpha, real sq_rho, real delta) {
    int N = dims(x)[1];
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = sq_alpha + delta;
      for (j in (i + 1):N) {
        K[i, j] = sq_alpha * exp(-sq_rho * square(x[i,j]) );
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = sq_alpha + delta;
    return K;
  }
}

data{
  array[10] int T;
  array[10] int society;
  array[10] int P;
  matrix[10,10] Dmat;
}

parameters{
 real<lower=0> a;
 real<lower=0> b;
 real<lower=0> etasq;
 real<lower=0> g;
 real<lower=0> rhosq;
 vector[10] k;
}

model{
  vector[10] lambda;
  matrix[10,10] SIGMA;
  rhosq ~ exponential(0.5);
  etasq ~ exponential(2);
  a ~ exponential(1);
  b ~ exponential(1);
  g ~ exponential(1);

  SIGMA = cov_GPL2(Dmat, etasq, rhosq, 0.01);
  k ~ multi_normal(rep_vector(0,10), SIGMA);
  for (i in 1:10) {
    lambda[i] = (a * P[i]^b/g) * exp(k[society[i]]);
  }
  T ~ poisson(lambda);
}
"""
data = {
    'T'      : m.df["total_tools"].values.astype(int),
    'P'      : m.df["population"].values.astype(int),
    'society': np.array(m.data_on_model['society'] + 1).astype(int),
    'Dmat'   : np.array(islandsDistMatrix),
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data, chains=4)

# 4. Comparison
param_map = {'a': 'a', 'b': 'b', 'g': 'g', 'etasq': 'etasq', 'rhosq': 'rhosq'}
plot_comparaison(m, df_stan, param_map, model_name=model_name)

# 5. Parameter Recovery Analysis
def estimate_rec(Dm_rec, P_rec, a_true, b_true, g_true, etasq_true, rhosq_true, society):
    n = Dm_rec.shape[0]
    K_sim = etasq_true * jnp.exp(-rhosq_true * jnp.square(Dm_rec))
    K_sim = K_sim.at[jnp.diag_indices(n)].add(0.01)
    K_sim = K_sim.astype(jnp.float32)

    m_rec = bf(print_devices_found=False, backend='tfp')
    k_sim = m_rec.dist.multivariate_normal(
        jnp.zeros(n, dtype=jnp.float32), K_sim, sample=True
    )

    lambda_sim = a_true * P_rec**b_true / g_true * jnp.exp(k_sim[society])
    total_tools_sim = m_rec.dist.poisson(lambda_sim.astype(jnp.float32), sample=True)

    m_rec.data_on_model = {
        'population' : jnp.array(P_rec, dtype=jnp.float32),
        'total_tools': jnp.array(total_tools_sim, dtype=jnp.int32),
        'Dmat'       : jnp.array(Dm_rec, dtype=jnp.float32),
        'society'    : jnp.array(society, dtype=jnp.int32),
    }

    def model_rec(Dmat, population, society, total_tools):
        a = yield m_rec.dist.exponential(1,   name='a')
        b = yield m_rec.dist.exponential(1,   name='b')
        g = yield m_rec.dist.exponential(1,   name='g')
        etasq = yield m_rec.dist.exponential(2,   name='etasq')
        rhosq = yield m_rec.dist.exponential(0.5, name='rhosq')

        nr = Dmat.shape[0]
        SIGMA = etasq * jnp.exp(-rhosq * jnp.square(Dmat))
        SIGMA = SIGMA.at[jnp.diag_indices(nr)].add(0.01)
        L_chol = jnp.linalg.cholesky(SIGMA)
        z = yield m_rec.dist.multivariate_normal(
            jnp.zeros(nr, dtype=jnp.float32), jnp.eye(nr, dtype=jnp.float32), name='k_z'
        )
        k = L_chol @ z

        lambda_ = a * population**b / g * jnp.exp(k[society])
        yield m_rec.dist.poisson(lambda_, obs=total_tools)

    m_rec.fit(model_rec, num_samples=1000, num_warmup=2000, progress_bar=False)
    sum_df = m_rec.summary()
    return sum_df.iloc[:, 0]


def param_recovery(nsim):
    results = []
    N_islands = 30  # more islands → better GP kernel identifiability

    for i in range(nsim):
        pos   = np.random.uniform(0, 10, size=(N_islands, 2))
        Dm_rec = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :])**2, axis=-1))
        P_rec  = np.exp(np.random.normal(10, 1.5, size=N_islands))
        society_rec = np.arange(N_islands)

        # Sample from priors with clipping to avoid degenerate lambda
        # np.random.exponential(scale) where scale = 1/rate
        a_true     = float(np.clip(np.random.exponential(1.0), 0.05, 5.0))
        b_true     = float(np.clip(np.random.exponential(1.0), 0.05, 1.0))
        g_true     = float(np.clip(np.random.exponential(1.0), 0.05, 5.0))
        etasq_true = float(np.clip(np.random.exponential(0.5), 0.1,  2.0))  # rate=2
        rhosq_true = float(np.clip(np.random.exponential(2.0), 0.1,  3.0))  # rate=0.5

        est = estimate_rec(Dm_rec, P_rec, a_true, b_true, g_true, etasq_true, rhosq_true, society_rec)

        def get_est(param):
            if param in est.index:        return float(est[param])
            if f"{param}[0]" in est.index: return float(est[f"{param}[0]"])
            return float('nan')

        param_map_rec = {
            'a'       : (a_true,               get_est('a')),
            'b'       : (b_true,               get_est('b')),
            'g'       : (g_true,               get_est('g')),
            'etasq'   : (etasq_true,           get_est('etasq')),
            'rhosq'   : (rhosq_true,           get_est('rhosq')),
            'a_over_g': (a_true / g_true,      get_est('a') / get_est('g')),
        }
        for param, (true_val, est_val) in param_map_rec.items():
            results.append({
                'sim'        : i,
                'parameter'  : param,
                'simulated'  : true_val,
                'estimations': est_val,
            })

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res


print("\nRunning Parameter Recovery...")
nsim = int(os.environ.get("BF_NSIM", 5))
res  = param_recovery(nsim=nsim)


# --- WAIC & LOO cross-check: native TFP vs NumPyro/ArviZ reference (same draws) ---
# `waic_ref` is a numpyro-mode transcription of the fitted TFP model (identical
# latent site names), used to evaluate the same posterior draws through
# numpyro.infer.log_likelihood -- the exact machinery ArviZ uses.
m_ref = bf(platform='cpu', print_devices_found=False)

def waic_ref(Dmat, population, society, total_tools):
    a = m_ref.dist.exponential(1)
    b = m_ref.dist.exponential(1)
    g = m_ref.dist.exponential(1)
    etasq = m_ref.dist.exponential(2)
    rhosq = m_ref.dist.exponential(0.5)
    n = Dmat.shape[0]
    SIGMA = etasq * jnp.exp(-rhosq * jnp.square(Dmat))
    SIGMA = SIGMA.at[jnp.diag_indices(n)].add(0.01)
    L_chol = jnp.linalg.cholesky(SIGMA)
    k_z = m_ref.dist.multivariate_normal(jnp.zeros(n, dtype=jnp.float32), jnp.eye(n, dtype=jnp.float32))
    k = L_chol @ k_z
    lambda_ = a * population**b / g * jnp.exp(k[society])
    m_ref.dist.poisson(lambda_, obs=total_tools)
waic_report(m, model_name, ref_model=waic_ref, ref_kwargs={'Dmat': m.data_on_model['Dmat'], 'population': m.data_on_model['population'], 'society': m.data_on_model['society'], 'total_tools': m.data_on_model['total_tools']})
