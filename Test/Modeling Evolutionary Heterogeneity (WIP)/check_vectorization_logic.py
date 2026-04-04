import jax
import jax.numpy as jnp
import numpy as np

# Original implementations (from fit_bi_ucln.py / fit_bi_gamma.py)
def get_hky_Q_orig(kappa, pi):
    Q = jnp.zeros((4,4))
    Q = Q.at[0,2].set(kappa * pi[2])
    Q = Q.at[2,0].set(kappa * pi[0])
    Q = Q.at[1,3].set(kappa * pi[3])
    Q = Q.at[3,1].set(kappa * pi[1])
    Q = Q.at[0,1].set(pi[1])
    Q = Q.at[0,3].set(pi[3])
    Q = Q.at[1,0].set(pi[0])
    Q = Q.at[1,2].set(pi[2])
    Q = Q.at[2,1].set(pi[1])
    Q = Q.at[2,3].set(pi[3])
    Q = Q.at[3,0].set(pi[0])
    Q = Q.at[3,2].set(pi[2])
    diag = -jnp.sum(Q, axis=1)
    for i in range(4):
        Q = Q.at[i,i].set(diag[i])
    return Q

def discrete_gamma_rates_orig(alpha, K=4):
    probs = jnp.linspace(0, 1, K + 1)[1:-1]
    z = jax.scipy.special.erfinv(2 * probs - 1) * jnp.sqrt(2)
    term1 = 1.0 - 1.0 / (9.0 * alpha)
    term2 = z / jnp.sqrt(9.0 * alpha)
    Q_internal = jnp.power(jnp.maximum(term1 + term2, 0.0), 3)
    cdf_internal = jax.scipy.special.gammainc(alpha + 1.0, alpha * Q_internal)
    cdf_all = jnp.concatenate([jnp.array([0.0]), cdf_internal, jnp.array([1.0])])
    rates = K * (cdf_all[1:] - cdf_all[:-1])
    return jnp.maximum(rates, 1e-6)

# Vectorized implementations (from BEAST_Algorithms.qmd)
_IS_TRANSITION = jnp.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=jnp.float32)

def get_hky_Q_vec(kappa, pi):
    Q_off = (1.0 + (kappa - 1.0) * _IS_TRANSITION) * (1.0 - jnp.eye(4)) * pi[None, :]
    return Q_off - jnp.diag(Q_off.sum(axis=1))

def discrete_gamma_rates_vec(alpha, K=4):
    probs   = jnp.linspace(0, 1, K + 1)[1:-1]
    z       = jax.scipy.stats.norm.ppf(probs)
    a9      = 9.0 * alpha
    Q_int   = jnp.maximum(1.0 - 1.0 / a9 + z / jnp.sqrt(a9), 0.0) ** 3
    cdf_int = jax.scipy.special.gammainc(alpha + 1.0, alpha * Q_int)
    cdf_all = jnp.concatenate([jnp.array([0.0]), cdf_int, jnp.array([1.0])])
    return jnp.maximum(K * jnp.diff(cdf_all), 1e-6)

# Test cases
kappas = [2.0, 5.0, 10.0]
alphas = [0.1, 0.5, 1.0, 5.0]
pi = jnp.array([0.3, 0.2, 0.1, 0.4])

print("Checking get_hky_Q numerical identity...")
for k in kappas:
    Q_o = get_hky_Q_orig(k, pi)
    Q_v = get_hky_Q_vec(k, pi)
    diff = jnp.abs(Q_o - Q_v).max()
    print(f"  kappa={k:.1f} | Max diff: {diff:.2e}")
    assert diff < 1e-6

print("\nChecking discrete_gamma_rates numerical identity...")
for a in alphas:
    R_o = discrete_gamma_rates_orig(a, K=4)
    R_v = discrete_gamma_rates_vec(a, K=4)
    diff = jnp.abs(R_o - R_v).max()
    print(f"  alpha={a:.1f} | Max diff: {diff:.2e}")
    assert diff < 1e-6

print("\nLogic verification: SUCCESS")
