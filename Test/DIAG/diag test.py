# %%
from patch_diag import patch_diag_class
from BI import bi

m = bi(platform="cpu")

# Import Data & Data Manipulation ------------------------------------------------
from importlib.resources import files

data_path = m.load.howell1(only_path=True)
m.data(data_path, sep=";")
m.df = m.df[m.df.age > 18]
m.scale(["weight"])


# Define model ------------------------------------------------
def model(weight, height):
    a = m.dist.normal(178, 20, name="a")
    b = m.dist.log_normal(0, 1, name="b")
    s = m.dist.uniform(0, 50, name="s")
    m.dist.normal(a + b * weight, s, obs=height)


# Run mcmc ------------------------------------------------
m.fit(model, progress_bar=False)

# %%
# --- m.summary() now uses JAX diagnostics by default ---
m.summary()

# %%
# --- ArviZ reference (old behaviour) ---
m.summary_old()

# %%
# --- standalone JAX summary ---
from jax_diagnostics import summary, rhat, ess, mcse, filter_posterior_dict

summary(m.posteriors_by_chain, group_by_chain=True)

# %%
# --- include: only show a and s ---
summary(m.posteriors_by_chain, group_by_chain=True, include=["a", "s"])

# %%
# --- exclude: hide b ---
summary(m.posteriors_by_chain, group_by_chain=True, exclude="b")

# %%
# --- filter_posterior_dict utility ---
# 'b' removed, 'b_other' would survive (base-name exact match)
sub = filter_posterior_dict(m.posteriors_by_chain, exclude="b")
print("after exclude b:", list(sub.keys()))

# %%
# --- m.filter_posteriors: modifies m.posteriors in-place, full preserved ---
m.filter_posteriors(exclude="b")
print("m.posteriors keys after filter:", list(m.posteriors.keys()))
print("m.posteriors_full keys (unchanged):", list(m.posteriors_full.keys()))

# %%
# --- reset by re-filtering from full ---
m.filter_posteriors(include=["a", "s"])
print("m.posteriors after include a,s:", list(m.posteriors.keys()))

# %%
# --- rhat / ess / mcse with include/exclude ---
print(rhat(m.posteriors_by_chain, include=["a", "b"]))
print(ess(m.posteriors_by_chain, include=["a"], kind="bulk"))
print(ess(m.posteriors_by_chain, exclude="b", kind="tail"))
print(mcse(m.posteriors_by_chain, include=["a", "s"]))
