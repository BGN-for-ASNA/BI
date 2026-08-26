"""BI translation of the coevolve-generated cichlid GDPM (Ringen 2026, §3.6).

Three Gaussian traits (Promiscuity, SpermSize, Predation) on a 265-tip cichlid
phylogeny, identity link, Lambda = I -- the smallest complete GDPM in the paper.

Correspondence with stan/cichlid_gdpm.stan:

  A_diag      vector<upper=0>[J] ~ std_normal   -> truncated_normal(0, 1, high=0)
  A_offdiag   vector[4]          ~ normal(0,2)  -> normal(0, 2)
  Q_sigma     vector<lower=0>[J] ~ normal(0,2)  -> truncated_normal(0, 2, low=0)
  b, eta_anc  vector[J]          ~ std_normal   -> normal(0, 1)
  z_drift     [N_seg-1, J]       ~ std_normal   -> normal(0, 1)
  likelihood  multi_normal_cholesky(y - eta_tip | 0, L_VCV_tip)

`terminal_drift` is deliberately not translated. In the Stan model it receives a
std_normal prior at every non-missing cell and is read only at missing cells, so
with complete data it is an independent block that factorises out of the joint
and cannot affect the posterior of any parameter compared here. The cichlid data
are simulated complete (miss is all zero, asserted below).
"""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = "/home/sebastian_sosa/phylo/examples/cichlid"
sys.path.insert(0, HERE + "/bf")
from gdpm_core import (ksolve, build_A, segment_quantities, traverse_levels,
                       tree_levels, tip_scale_tril)

from BayesForge import bf

m = bf(platform="cpu")


def load_data(path=HERE + "/data/standata.json"):
    D = json.load(open(path))
    assert D["N_tree"] == 1, "translation covers the single-tree case"
    miss = np.array(D["miss"])
    assert miss.sum() == 0, "terminal_drift was dropped; that needs complete data"

    eff = np.array(D["effects_mat"])
    J = D["J"]
    rows, cols = [], []
    for i in range(J):          # Stan ticker order: row-major, skipping diagonal
        for j in range(J):
            if i != j and eff[i, j] == 1:
                rows.append(i)
                cols.append(j)

    node_seq = np.array(D["node_seq"][0]) - 1        # Stan is 1-based
    parent = np.array(D["parent"][0]) - 1
    level_seg, level_valid = tree_levels(node_seq, parent, D["N_seg"])

    return dict(
        y=jnp.array(D["y"]),
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
        J=J,
    )


def make_model(N_seg, N_tips, J):
    def model(y, node_seq, parent, ts, tip, tip_id, off_rows, off_cols,
              level_seg, level_valid):
        # names given explicitly: BI's inference of the site name from the
        # assignment target does not fire for a model built inside a closure
        A_diag = m.dist.truncated_normal(0.0, 1.0, high=0.0, shape=(J,),
                                         name="A_diag")
        A_offdiag = m.dist.normal(0.0, 2.0, shape=(off_rows.shape[0],),
                                  name="A_offdiag")
        Q_sigma = m.dist.truncated_normal(0.0, 2.0, low=0.0, shape=(J,),
                                          name="Q_sigma")
        b = m.dist.normal(0.0, 1.0, shape=(J,), name="b")
        eta_anc = m.dist.normal(0.0, 1.0, shape=(J,), name="eta_anc")
        z_drift = m.dist.normal(0.0, 1.0, shape=(N_seg - 1, J), name="z_drift")

        A = build_A(A_diag, A_offdiag, off_rows, off_cols)
        Q = jnp.diag(Q_sigma ** 2)
        Q_inf = ksolve(A, Q)
        m.dist.track("A", A)
        m.dist.track("Q", Q)

        A_delta, L_VCV, A_solve = segment_quantities(A, Q_inf, ts)
        eta = traverse_levels(eta_anc, z_drift, A_delta, L_VCV, A_solve, b,
                              node_seq, parent, tip, level_seg, level_valid)
        L_tips = tip_scale_tril(L_VCV, node_seq, tip, N_tips)

        # tdrift = y - eta at the tip; Stan scores it under MVN(0, L_VCV_tip)
        resid = y - eta[tip_id]
        m.dist.multivariate_normal(loc=jnp.zeros(J), scale_tril=L_tips[tip_id],
                                   obs=resid, name="resid")

    return model


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--out", default=HERE + "/results/bf_draws.npz")
    args = p.parse_args()

    d = load_data()
    model = make_model(d["N_seg"], d["N_tips"], d["J"])
    obs = {k: d[k] for k in ("y", "node_seq", "parent", "ts", "tip", "tip_id",
                             "off_rows", "off_cols", "level_seg", "level_valid")}

    m.fit(model=model, obs=obs, num_warmup=args.warmup, num_samples=args.samples,
          num_chains=args.chains, target_accept_prob=0.95, seed=42)

    print(m.summary(var_names=["A", "Q", "b", "eta_anc", "A_diag", "A_offdiag",
                               "Q_sigma"]))

    post = m.posteriors_full if isinstance(m.posteriors_full, dict) else m.posteriors
    keep = {k: np.asarray(v) for k, v in post.items()
            if k in ("A", "Q", "b", "eta_anc", "A_diag", "A_offdiag", "Q_sigma")}
    np.savez(args.out, **keep)
    print("saved", args.out, {k: v.shape for k, v in keep.items()})
