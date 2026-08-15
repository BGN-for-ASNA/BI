# %%
from patch_diag import patch_diag_class
from BayesForge import bf

m = bf(platform="cpu")

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
# ---Filtering logic ---
m.summary(exclude="s")
# %%
# --- posterior change ---
m.posteriors.keys()
# %%
# --- posterior change ---
m.posteriors_full.keys()
# %%
m.diag.forest()
# %%
m.diag.density()
# %%
