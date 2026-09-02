"""BayesForge variant of the primate GDPM using mask= for missing data.

Identical to primates_bi.py except that the two partly-missing traits are
handled with mask= rather than index-subsetting. Used to verify the
np_dists.py mask fix end to end: its posterior must match the index-subset
run and the Stan reference.

Original header follows.

BI translation of stan/GDPM_primate.stan (Ringen 2026, §3.7).

The empirical primate model. Two latent traits -- life history (eta_1) and
brain allometry (eta_2) -- coevolve on a 143-tip consensus tree and are mapped
by a factor matrix Lambda onto four gamma-distributed observed traits, two of
which have missing values.

What is new relative to the cichlid translation:

  Q from a correlation Cholesky   L_R ~ lkj_corr_cholesky(4)
                                  Q = diag(Q_sigma) L_R L_R' diag(Q_sigma)
  factor matrix Lambda            fixed 1s plus two free loadings
  gamma observation model         gamma(shape, shape/exp(mu)) per trait
  brain allometry                 beta = softplus(eta_2), mu = alpha + beta*log(body)
  terminal_drift                  a real parameter here, scored under
                                  MVN(0, L_VCV_tip) and added to eta at the tips
  missing data                    longevity/maturity are subset to their
                                  observed rows, not masked (see load_data)

The scaling by y_mean, and the fact that brain is modelled on the *unscaled*
body value, follow the Stan source exactly.
"""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = "/home/sebastian_sosa/phylo/examples/primates"
sys.path.insert(0, "/home/sebastian_sosa/phylo/examples")
from gdpm_core import (ksolve, build_A, segment_quantities, traverse_levels,
                       tree_levels, tip_scale_tril)

from BayesForge import bf

m = bf(platform="cpu")


def load_data(path=HERE + "/data/standata.json"):
    D = json.load(open(path))
    assert D["N_tree"] == 1, "translation covers the single-tree case"
    K = D["N_latent"]

    eff = np.array(D["effects_mat"])
    rows, cols = [], []
    for i in range(K):          # Stan ticker order: row-major, skipping diagonal
        for j in range(K):
            if i != j and eff[i, j] == 1:
                rows.append(i)
                cols.append(j)

    node_seq = np.array(D["node_seq"][0]) - 1        # Stan is 1-based
    parent = np.array(D["parent"][0]) - 1
    level_seg, level_valid = tree_levels(node_seq, parent, D["N_seg"])

    y = np.array(D["y"], dtype=float)
    miss = np.array(D["miss"], dtype=float)
    # -99 placeholders would make the gamma log-density NaN, so park them on a
    # harmless positive value. They are never scored: the two partly-missing
    # traits are subset to their observed rows below.
    y_safe = np.where(miss == 1, 1.0, y)

    # Stan guards each partly-missing trait with `if (miss[i,j] == 0)`. We take
    # the same subset explicitly rather than masking: BI forwards `mask=` to
    # numpyro's `obs_mask` inference hint, which did NOT exclude these rows --
    # the placeholders entered the likelihood and collapsed shape[3]/shape[4]
    # (0.98 vs Stan's 17.9). Indexing is unambiguous.
    obs_idx = {j: jnp.array(np.where(miss[:, j] == 0)[0]) for j in (2, 3)}

    return dict(
        y=jnp.array(y_safe),
        observed=jnp.array(miss == 0),
        idx_longevity=obs_idx[2],
        idx_maturity=obs_idx[3],
        y_mean=jnp.array(D["y_mean"]),
        node_seq=jnp.array(node_seq),
        parent=jnp.array(parent),
        ts=jnp.array(D["ts"][0]),
        tip=jnp.array(D["tip"][0]),
        tip_id=jnp.array(np.array(D["tip_id"]) - 1),
        off_rows=jnp.array(rows),
        off_cols=jnp.array(cols),
        level_seg=level_seg,
        level_valid=level_valid,
        N_tips=D["N_tips"],
        N_seg=D["N_seg"],
        N_obs=D["N_obs"],
        J=D["J"],
        K=K,
    )


def make_model(N_seg, N_tips, N_obs, J, K):
    def model(y, observed, idx_longevity, idx_maturity, y_mean, node_seq,
              parent, ts, tip, tip_id, off_rows, off_cols,
              level_seg, level_valid):
        A_diag = m.dist.truncated_normal(0.0, 1.0, high=0.0, shape=(K,),
                                         name="A_diag")
        A_offdiag = m.dist.normal(0.0, 1.0, shape=(off_rows.shape[0],),
                                  name="A_offdiag")
        Q_sigma = m.dist.truncated_normal(0.0, 1.0, low=0.0, shape=(K,),
                                          name="Q_sigma")
        L_R = m.dist.lkj_cholesky(K, concentration=4.0, name="L_R")
        b = m.dist.normal(0.0, 1.0, shape=(K,), name="b")
        # informative prior on the ancestral brain-allometry state, latent scale
        eta_anc = m.dist.normal(jnp.array([0.0, -0.2]), jnp.array([1.0, 0.15]),
                                name="eta_anc")
        z_drift = m.dist.normal(0.0, 1.0, shape=(N_seg - 1, K), name="z_drift")
        alpha = m.dist.normal(0.0, 1.0, shape=(J,), name="alpha")
        shape_ = m.dist.gamma(0.01, 0.01, shape=(J,), name="shape")
        lambda_free = m.dist.normal(0.0, 1.0, shape=(J - K,), name="lambda_free")

        A = build_A(A_diag, A_offdiag, off_rows, off_cols)
        LQ = Q_sigma[:, None] * L_R
        Q = LQ @ LQ.T
        Q_inf = ksolve(A, Q)
        m.dist.track("A", A)
        m.dist.track("Q", Q)
        m.dist.track("cor_R", L_R @ L_R.T)

        # GDPM_primate.stan is hand-written and does NOT symmetrise
        # A^-1(A_delta - I); only coevolve's generated code does. A_solve
        # multiplies b, so symmetrising here shifts b and A[1,2] off the
        # Stan reference (JSD 0.19 / 0.11 before this was corrected).
        A_delta, L_VCV, A_solve = segment_quantities(
            A, Q_inf, ts, symmetrize_A_solve=False)
        eta = traverse_levels(eta_anc, z_drift, A_delta, L_VCV, A_solve, b,
                              node_seq, parent, tip, level_seg, level_valid)
        L_tips = tip_scale_tril(L_VCV, node_seq, tip, N_tips)

        # Lambda: [0,0] and [1,1] fixed to 1, two free loadings on eta_1
        Lambda = jnp.zeros((J, K))
        Lambda = Lambda.at[0, 0].set(1.0).at[1, 1].set(1.0)
        Lambda = Lambda.at[2, 0].set(lambda_free[0]).at[3, 0].set(lambda_free[1])
        m.dist.track("Lambda", Lambda)

        terminal_drift = m.dist.multivariate_normal(
            loc=jnp.zeros(K), scale_tril=L_tips[tip_id], shape=(N_obs,),
            name="terminal_drift")
        eta_tips = eta[tip_id] + terminal_drift

        # body size
        mu1 = alpha[0] + Lambda[0, 0] * eta_tips[:, 0]
        m.dist.gamma(shape_[0], shape_[0] / jnp.exp(mu1),
                     obs=y[:, 0] / y_mean[0], name="obs_body")
        # brain size: allometric slope on the unscaled body value
        beta = jax.nn.softplus(Lambda[1, 1] * eta_tips[:, 1])
        mu2 = alpha[1] + jnp.log(y[:, 0]) * beta
        m.dist.gamma(shape_[1], shape_[1] / jnp.exp(mu2),
                     obs=y[:, 1], name="obs_brain")
        # longevity and maturity via mask= -- this is the construct under test.
        # Before the np_dists.py fix the mask was inert and the y_safe
        # placeholders were scored, collapsing shape[3]/shape[4].
        mu3 = alpha[2] + Lambda[2, 0] * eta_tips[:, 0]
        m.dist.gamma(shape_[2], shape_[2] / jnp.exp(mu3),
                     obs=y[:, 2] / y_mean[2], mask=observed[:, 2],
                     name="obs_longevity")
        mu4 = alpha[3] + Lambda[3, 0] * eta_tips[:, 0]
        m.dist.gamma(shape_[3], shape_[3] / jnp.exp(mu4),
                     obs=y[:, 3] / y_mean[3], mask=observed[:, 3],
                     name="obs_maturity")

    return model


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--out", default=HERE + "/results/bf_mask_draws.npz")
    args = p.parse_args()

    d = load_data()
    model = make_model(d["N_seg"], d["N_tips"], d["N_obs"], d["J"], d["K"])
    keys = ("y", "observed", "idx_longevity", "idx_maturity", "y_mean",
            "node_seq", "parent", "ts", "tip", "tip_id", "off_rows", "off_cols",
            "level_seg", "level_valid")
    obs = {k: d[k] for k in keys}

    m.fit(model=model, obs=obs, num_warmup=args.warmup, num_samples=args.samples,
          num_chains=args.chains, target_accept_prob=0.95, seed=42)

    want = ["A", "Q", "cor_R", "b", "eta_anc", "A_diag", "A_offdiag", "Q_sigma",
            "alpha", "shape", "lambda_free"]
    print(m.summary(var_names=want))

    post = m.posteriors_full
    keep = {k: np.asarray(v) for k, v in post.items() if k in want}
    np.savez(args.out, **keep)
    print("saved", args.out, {k: v.shape for k, v in keep.items()})
