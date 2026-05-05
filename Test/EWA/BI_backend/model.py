import jax
import jax.numpy as jnp


def make_model(m):
    """Return bi_ewa_model closure bound to BI instance m."""

    def bi_ewa_model(K, J, tech, id, bout, y_prev, s, ps, ks, pr, co, yo, age):
        lam   = m.dist.exponential(name="lambda", rate=1.0)
        mu    = m.dist.normal(name="mu", loc=jnp.zeros(8), scale=1.0)
        sigma = m.dist.exponential(name="sigma", rate=jnp.full(8, 3.0))
        L_Rho = m.dist.lkj_cholesky(name="L_Rho", dimension=8, concentration=3.0)
        z     = m.dist.normal(name="z", loc=jnp.zeros((8, J)), scale=1.0)
        b_age = m.dist.normal(name="b_age", loc=jnp.zeros(2), scale=1.0)

        L_scaled = L_Rho * sigma[:, None]
        a_id     = (L_scaled @ z).T

        phi_arr   = jax.nn.sigmoid(mu[0] + a_id[id, 0] + b_age[0] * age)
        gamma_arr = jax.nn.sigmoid(mu[1] + a_id[id, 1] + b_age[1] * age)
        fconf_arr = jnp.exp(mu[2] + a_id[id, 2])
        B_pay     = mu[3] + a_id[id, 3]
        B_kin     = mu[4] + a_id[id, 4]
        B_pres    = mu[5] + a_id[id, 5]
        B_coho    = mu[6] + a_id[id, 6]
        B_yob     = mu[7] + a_id[id, 7]

        def step_fn(carry, x):
            AC, ll = carry
            (phi_i, gamma_i, fconf_i, Bp, Bk, Bpr, Bc, By,
             t_i, id_i, b_i, py_i, s_i, ps_i, ks_i, pr_i, co_i, yo_i) = x

            ac_new = jnp.where(b_i == 1, jnp.zeros(K), (1.0 - phi_i) * AC[id_i] + phi_i * py_i)
            logPrA = (lam * ac_new)[t_i] - jax.nn.logsumexp(lam * ac_new)

            lin_rest   = jnp.exp(Bp * ps_i[1:] + Bk * ks_i[1:] + Bpr * pr_i[1:] + Bc * co_i[1:] + By * yo_i[1:])
            lin_mod    = jnp.concatenate([jnp.ones(1), lin_rest]) * jnp.power(s_i, fconf_i)
            safe_denom = jnp.where(jnp.sum(lin_mod) > 0, jnp.sum(lin_mod), 1.0)
            PrS        = lin_mod[t_i] / safe_denom

            prob_mix   = (1.0 - gamma_i) * jnp.exp(logPrA) + gamma_i * PrS
            social_obs = jnp.logical_and(b_i > 1, jnp.sum(s_i) > 0)
            lik_i      = jnp.where(social_obs, jnp.log(prob_mix), logPrA)
            return (AC.at[id_i].set(ac_new), ll + lik_i), None

        AC_init = jnp.zeros((J, K))
        xs = (phi_arr, gamma_arr, fconf_arr, B_pay, B_kin, B_pres, B_coho, B_yob,
              tech, id, bout, y_prev, s, ps, ks, pr, co, yo)
        (_, total_ll), _ = jax.lax.scan(step_fn, (AC_init, 0.0), xs)
        m.dist.unit(total_ll, name="obs")

    return bi_ewa_model
