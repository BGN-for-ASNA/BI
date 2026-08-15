import sys
import os
import time
import jax
import jax.numpy as jnp
from BayesForge import BayesForge
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import psutil

def rate_matrix(h, q):
    """
    Vectorized construction of the (S+1)×(S+1) continuous-time rate matrix Q.
    h: [S]        mortality hazard rates (state S+1 = dead, absorbing)
    q: [S*(S-1)]  transition rates among alive states (diagonal excluded)

    Strategy
    --------
    1. Reshape q to (S, S-1): each row holds the S-1 rates leaving state s.
    2. Map compact column indices → full column indices via broadcast arithmetic
       (no Python loops, fully JIT-friendly).
    3. Zero the diagonal, then set it to -(row_sum + h_s).
    4. Pad to (S+1, S+1): insert mortality column + absorbing dead row.
    """
    S   = h.shape[0]
    Sp1 = S + 1

    q_mat = q.reshape(S, S - 1)          # (S, S-1)

    # Broadcast index grids
    c = jnp.arange(S)[None, :]           # (1, S)  – full column indices
    r = jnp.arange(S)[:, None]           # (S, 1)  – row indices

    # Map each full column k to its compact index in q_mat[s]:
    #   k < s  →  compact index = k
    #   k > s  →  compact index = k - 1
    #   k == s →  diagonal (handled separately)
    q_col = jnp.where(c < r, c, c - 1)  # (S, S)

    # Gather transition rates; diagonal entries carry a garbage index (k-1==k),
    # but are zeroed out in the next step.
    Q_alive = q_mat[r, q_col]            # (S, S)
    Q_alive = Q_alive * (1 - jnp.eye(S))  # zero the diagonal

    # Diagonal = -(sum of off-diagonal row rates + mortality)
    diag_vals = -(Q_alive.sum(axis=1) + h)   # (S,)
    Q_alive = Q_alive.at[jnp.arange(S), jnp.arange(S)].set(diag_vals)

    # Assemble full (S+1)×(S+1) matrix
    Q = jnp.zeros((Sp1, Sp1))
    Q = Q.at[:S, :S].set(Q_alive)   # alive-to-alive block
    Q = Q.at[:S,  S].set(h)         # mortality column
    # Dead row stays zero (absorbing state)
    return Q

def forward_algorithm_vectorized(y, f, l, log_H, logit_p):
    """
    Marginalize out discrete states using a masked vectorized forward algorithm.
    log_H: [J-1, Sp1, Sp1] log transition matrices (targets survey j from j-1)
    logit_p: [J, S] detection logits for alive states
    """
    N, J = y.shape
    Sp1 = log_H.shape[1]
    S = Sp1 - 1
    
    log_p_alive = jax.nn.log_sigmoid(logit_p)
    log_q_alive = jax.nn.log_sigmoid(-logit_p)
    
    def individual_ll(y_i, f_i, l_i):
        f_idx = f_i - 1
        observed_state = y_i[f_idx] - 1
        # alpha_init at time f_idx
        alpha_init = jnp.full((Sp1,), -1e10)
        alpha_init = alpha_init.at[observed_state].set(0.0)
        
        def step(alpha, val):
            j, y_ij = val
            # Transition to survey j (process transition j-1 -> j using log_H[j-1])
            alpha_next = jax.scipy.special.logsumexp(alpha[:, None] + log_H[j - 1], axis=0)
            
            # Observation at survey j
            is_seen = y_ij > 0
            seen_state = y_ij - 1
            
            # Case Seen: only the seen state is possible
            log_p_seen = jnp.full((Sp1,), -1e10)
            log_p_seen = log_p_seen.at[seen_state].set(log_p_alive[j, seen_state])
            
            # Case Not Seen: all states possible with (1-p) or 1
            log_p_not_seen = jnp.zeros((Sp1,))
            log_p_not_seen = log_p_not_seen.at[:S].set(log_q_alive[j])
            log_p_not_seen = log_p_not_seen.at[S].set(0.0)
            
            log_obs = jnp.where(is_seen, log_p_seen, log_p_not_seen)
            
            # Only update if survey index j > f_idx
            update = j > f_idx
            alpha = jnp.where(update, alpha_next + log_obs, alpha)
            return alpha, None

        # Process all surveys 0..J-1. Updates only happen AFTER survey f_idx.
        # Survey 0 has no preceding transition in this model (CJS starts at first capture).
        final_alpha, _ = jax.lax.scan(step, alpha_init, (jnp.arange(J), y_i), reverse=False)
        return jax.scipy.special.logsumexp(final_alpha)

    return jax.vmap(individual_ll)(y, f, l)

def BF_mscr_model(y, f, l, tau, n_states, n_surveys, n_individuals):
    jax.debug.print("y.shape: {x}", x=y.shape)
    m = BF()
    S = n_states
    J = n_surveys
    Jm1 = J - 1
    
    # Priors (match Stan exactly)
    h = m.dist.gamma(1, 3, shape=(S,), name='h')
    q = m.dist.gamma(1, 3, shape=(S * (S-1),), name='q')
    p_detect = m.dist.beta(1, 1, shape=(S, Jm1), name='p_detect')
    
    # Tau scaling (match Stan: tau_scl = tau / exp(mean(log(tau))))
    tau_scl = tau / jnp.exp(jnp.mean(jnp.log(tau)))
    
    # Clip priors for stability during sampling
    h_cl = jnp.clip(h, 0, 100)
    q_cl = jnp.clip(q, 1e-6, 100)
    
    # Build transition matrices
    Q = rate_matrix(h_cl, q_cl)
    # Apply scaled tau
    Q_tau = Q[None, :, :] * tau_scl[:, None, None]
    H = jax.vmap(jax.scipy.linalg.expm)(Q_tau)
    
    # Numerical stability: clip H and avoid log(0)
    H_cl_mat = jnp.clip(H, a_min=1e-15, a_max=1.0)
    log_H = jnp.log(H_cl_mat)
    
    # Detection logits
    logit_p = jnp.full((J, S), -1e10) # Default to 0 prob for survey 1
    logit_p = logit_p.at[1:, :].set(jax.scipy.special.logit(jnp.clip(p_detect.T, 1e-6, 1-1e-6)))
    
    log_liks = forward_algorithm_vectorized(y, f, l, log_H, logit_p)
    
    # Guard against NaN/Inf in factor
    total_log_lik = jnp.sum(jnp.where(jnp.isfinite(log_liks), log_liks, -1e8))
    jax.debug.print("total_log_lik: {x}", x=total_log_lik)
    total_log_lik = jnp.clip(total_log_lik, a_min=-1e12, a_max=1e6)
    numpyro.factor("obs", total_log_lik)

def run_bi_benchmark(n_individuals=400):
    from prepare_data import load_fleayi_data
    # Use 3 states for fleayi (Healthy, Sick, Very Sick - or whatever the indices are)
    data = load_fleayi_data(n_individuals)
    
    m = BF(platform='cpu')
    
    start_time = time.time()
    
    # Stan means from N=400 run (3 states)
    stan_h = jnp.array([2.58, 0.35, 1.10])
    stan_q = jnp.array([1.14, 1.07, 1.71, 0.45, 0.54, 1.14])
    
    init_values = {
        'h': stan_h,
        'q': stan_q,
        'p_detect': jnp.full((3, 20), 0.5)
    }
    
    m.fit(
        model=BF_mscr_model,
        obs=data,
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        seed=42,
        init_strategy=numpyro.infer.init_to_value(values=init_values)
    )
    
    exec_time = time.time() - start_time
    
    # Extract means for parity check
    posteriors = m.posteriors
    params = ['h', 'q']
    for p_name in params:
        mean_val = jnp.mean(posteriors[p_name], axis=0)
        print(f"Posterior mean {p_name}: {mean_val}")
    
    return {"time": exec_time, "h_mean": jnp.mean(posteriors['h'], axis=0), "q_mean": jnp.mean(posteriors['q'], axis=0)}

if __name__ == "__main__":
    n = 400
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    results = run_bi_benchmark(n)
    print(f"BF N={n} Time: {results['time']:.2f}s")
