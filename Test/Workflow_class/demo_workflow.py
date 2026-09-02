"""Usage demo for BayesForge.Workflow (``m.workflow``).

Not a pytest suite -- a runnable walkthrough of every Workflow method, in the
same spirit as the other scripts under Test/. Requires the full BF stack
(jax/numpyro) to actually execute; it was syntax-checked with
``python -m py_compile`` but not run in the environment this was written in
(no jax installed there).

Run with: python "Test/Workflow_class/demo_workflow.py"
"""
from BayesForge import bf
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 0. Confirm what parallelization is actually available
# ---------------------------------------------------------------------------
m = bf(platform="cpu")
m.workflow.parallel_report()


# ---------------------------------------------------------------------------
# 1. Model + DGP, bound to this `m` (the n_jobs=1 path)
# ---------------------------------------------------------------------------
def model(x, y):
    alpha = m.dist.normal(0, 10, name="alpha")
    beta = m.dist.normal(0, 1, name="beta")
    sigma = m.dist.exponential(1, name="sigma")
    m.dist.normal(alpha + beta * x, sigma, obs=y)


def dgp(n_obs=200):
    alpha = m.dist.normal(0, 10, sample=True)
    beta = m.dist.normal(0, 1, sample=True)
    sigma = m.dist.exponential(1, sample=True)
    x = m.dist.normal(0, 1, sample=True, shape=(n_obs,))   # random X: observational DGP
    y = m.dist.normal(alpha + beta * x, sigma, sample=True)
    return dict(alpha=alpha, beta=beta, sigma=sigma), dict(x=x, y=y)


# ---------------------------------------------------------------------------
# 2. Advice before spending any compute
# ---------------------------------------------------------------------------
_, sample_data = dgp()
for line in m.workflow.advise(data=sample_data, n_params=3, dgp=dgp, model=model):
    print("-", line)


# ---------------------------------------------------------------------------
# 3. Parameter recovery (sequential, n_jobs=1)
# ---------------------------------------------------------------------------
recovery = m.workflow.recover(
    model=model, dgp=dgp, param_names=["alpha", "beta", "sigma"],
    n_sim=20,  # keep small for a quick demo; use >=100 for a real check
    fit_kwargs=dict(num_warmup=500, num_samples=500, num_chains=4),
    results_dir="Results/demo_recovery",
)
print(recovery.summary())
m.workflow.plot_recovery(recovery)  # .show() in an interactive session


# ---------------------------------------------------------------------------
# 4. Simulation-based calibration checking
# ---------------------------------------------------------------------------
sbc_result = m.workflow.sbc(
    model=model, dgp=dgp, param_names=["alpha", "beta"],
    n_sbc=50, n_post_draws=500,  # keep small for a quick demo; use >=200 for real SBC
    results_dir="Results/demo_sbc",
)
print(sbc_result.summary())
m.workflow.plot_sbc(sbc_result)


# ---------------------------------------------------------------------------
# 5. Fit once on real-shaped data, then use the diagnostics/posterior-arithmetic side
# ---------------------------------------------------------------------------
_, data = dgp(n_obs=200)
m.fit(model, obs=data, num_warmup=1000, num_samples=1000, num_chains=4)

annotated = m.workflow.annotated_summary()
print(annotated[["mean", "sd", "r_hat", "ess_bulk", "ess_tail", "verdict", "interpretation"]])
m.workflow.plot_annotated_summary(annotated)

contrast = m.workflow.contrast("beta", name="slope")
print(contrast.summary())

decision = m.workflow.decide(
    utility_fn=lambda outcome: jnp.where(outcome > 0, 10.0, -1.0),
    actions={
        "treat":   m.posteriors["alpha"] + m.posteriors["beta"] * 1.0,
        "control": m.posteriors["alpha"] + m.posteriors["beta"] * 0.0,
    },
)
print(decision)


# ---------------------------------------------------------------------------
# 6. m.dgp: the DGP used above round-trips through m.save()/m.load()
# ---------------------------------------------------------------------------
# workflow.recover()/sbc() already persisted `dgp` onto m.dgp automatically
# (step 3/4 above). It's a plain attribute, so cloudpickle carries it through
# save/load exactly like m.model, m.posteriors, m.data_on_model:
assert m.dgp is dgp

m.save("Results/demo_model.pkl")
m2 = bf.load("Results/demo_model.pkl")
true_params_again, data_again = m2.dgp()          # the exact DGP survived the round trip
print("Reloaded m2.dgp() ->", true_params_again)

# Once m.dgp/m.model are set (directly, via a prior recover()/sbc() call, or
# a reload), later calls don't need to pass model=/dgp= again -- note we do
# NOT pass the original `model`/`dgp` variables here: those closures capture
# the *original* m, whereas m2.model/m2.dgp were correctly rebound to m2 by
# cloudpickle's self-referential save/load (the same mechanism that already
# lets self.model round-trip):
more_recovery = m2.workflow.recover(
    param_names=["alpha", "beta", "sigma"], n_sim=5,
    fit_kwargs=dict(num_warmup=200, num_samples=200, num_chains=2),
)
print(more_recovery.summary())


# ---------------------------------------------------------------------------
# 7. n_jobs>1 path -- requires factories (see Workflow.recover docstring for why)
# ---------------------------------------------------------------------------
def model_factory(m_):
    def _model(x, y):
        alpha = m_.dist.normal(0, 10, name="alpha")
        beta = m_.dist.normal(0, 1, name="beta")
        sigma = m_.dist.exponential(1, name="sigma")
        m_.dist.normal(alpha + beta * x, sigma, obs=y)
    return _model


def dgp_factory(m_):
    def _dgp(n_obs=200):
        alpha = m_.dist.normal(0, 10, sample=True)
        beta = m_.dist.normal(0, 1, sample=True)
        sigma = m_.dist.exponential(1, sample=True)
        x = m_.dist.normal(0, 1, sample=True, shape=(n_obs,))
        y = m_.dist.normal(alpha + beta * x, sigma, sample=True)
        return dict(alpha=alpha, beta=beta, sigma=sigma), dict(x=x, y=y)
    return _dgp


if __name__ == "__main__":
    parallel_recovery = m.workflow.recover(
        model_factory=model_factory, dgp_factory=dgp_factory,
        param_names=["alpha", "beta", "sigma"], n_sim=20, n_jobs=4,
        fit_kwargs=dict(num_warmup=500, num_samples=500, num_chains=2),
    )
    print(parallel_recovery.summary())
