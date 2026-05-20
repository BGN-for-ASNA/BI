# %%
import sys
import re
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import cmdstanpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from BI import bi
from BI.Resources.datasets import load as bi_load

# Load BI/model.py as a module without making BI/ a package
_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "bi_model", _here / "BI_backend" / "model.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_model = _mod.make_model

# 1. Load data via BI loader
data_path = bi_load().panama_ewa(only_path=True)
d = pd.read_csv(data_path)

for col_prefix in ["Yobs", "cos", "ks", "ps", "prs"]:
    cols = [f"{col_prefix}{i}" for i in range(1, 8)]
    max_val = d[cols].max().max()
    d[cols] = d[cols] / max_val

stan_data = {
    "K": 7,  # Number of foraging techniques (options)
    "N": len(d),  # Number of observations (rows)
    "J": d["mono_index"].nunique(),  # Number of unique individuals (monkeys)
    "tech": d["tech_index"].values.astype(
        int
    ),  # Technique chosen at each observation (1-indexed)
    "y": d[
        [f"y{i}" for i in range(1, 8)]
    ].values,  # Observed personal yields for each of the K techniques (N×K)
    "s": d[
        [f"s{i}" for i in range(1, 8)]
    ].values,  # Frequency cue: number of times each technique was observed socially (N×K)
    "ps": d[
        [f"ps{i}" for i in range(1, 8)]
    ].values,  # Payoff-bias cue: observed payoff from social models for each technique (N×K)
    "ks": d[
        [f"ks{i}" for i in range(1, 8)]
    ].values,  # Kin-bias cue: whether demonstrator is matrilineal kin (N×K)
    "press": d[
        [f"prs{i}" for i in range(1, 8)]
    ].values,  # Rank-bias cue: whether demonstrator is alpha male/female (N×K)
    "cohos": d[
        [f"cos{i}" for i in range(1, 8)]
    ].values,  # Cohort-bias cue: age-similarity between focal and demonstrator (N×K)
    "yobs": d[
        [f"Yobs{i}" for i in range(1, 8)]
    ].values,  # Age-bias (YOB) cue: year-of-birth prestige bias (N×K)
    "bout": d["forg_bout"].values.astype(
        int
    ),  # Foraging bout index (1 = first bout for an individual, resets attraction scores)
    "id": d["mono_index"].values.astype(int),  # Individual (monkey) ID (1-indexed)
    "N_effects": 8,  # Number of learning parameters to estimate (phi, gamma, fconf, + 5 social cues)
    "age": d["age.c"].values,  # Centered age of the monkey at each observation
}
# %%
# 2. Run Stan model
print("Running Stan model...")
stan_src = "Stan/PN_social_global_age.stan"
stan_tmp = "Stan/temp_PN_social_global_age.stan"

with open(stan_src) as f:
    content = f.read()

content = content.replace("#//", "//").replace("#", "//")
content = re.sub(r"generated quantities\s*\{.*\}", "", content, flags=re.DOTALL)
content = re.sub(
    r"(int|real)\s+([a-zA-Z0-9_]+)\[([^\]]+)\];", r"array[\3] \1 \2;", content
)

with open(stan_tmp, "w") as f:
    f.write(content)

sm = cmdstanpy.CmdStanModel(stan_file=stan_tmp)
fit_stan = sm.sample(
    data=stan_data,
    iter_sampling=1000,
    iter_warmup=1000,
    chains=2,
    parallel_chains=2,
    adapt_delta=0.9,
)

stan_summary = fit_stan.summary()
stan_params = [
    "lambda",
    "mu[1]",
    "mu[2]",
    "mu[3]",
    "mu[4]",
    "mu[5]",
    "mu[6]",
    "mu[7]",
    "mu[8]",
    "b_age[1]",
    "b_age[2]",
    "sigma[1]",
    "sigma[2]",
    "sigma[3]",
    "sigma[4]",
    "sigma[5]",
    "sigma[6]",
    "sigma[7]",
    "sigma[8]",
]
stan_means = {p: stan_summary.loc[p, "Mean"] for p in stan_params}
print("Stan means:", stan_means)

# 3. Run BI model
print("Running BI model...")

y_prev = np.zeros_like(stan_data["y"])
y_prev[1:] = stan_data["y"][:-1]

m = bi(platform="cpu")
bi_ewa_model = make_model(m)

bi_data = {
    "K": 7,  # Number of foraging techniques (options)
    "J": stan_data["J"],  # Number of unique individuals (monkeys)
    "tech": stan_data["tech"] - 1,  # Technique chosen (0-indexed for JAX)
    "id": stan_data["id"] - 1,  # Individual (monkey) ID (0-indexed for JAX)
    "bout": stan_data[
        "bout"
    ],  # Foraging bout index (1 = first bout, resets attraction scores)
    "y_prev": y_prev,  # Previous personal yields, lagged by one row (N×K); used to update attraction scores
    "s": stan_data["s"],  # Frequency cue: social observation counts per technique (N×K)
    "ps": stan_data["ps"],  # Payoff-bias cue: social payoff per technique (N×K)
    "ks": stan_data[
        "ks"
    ],  # Kin-bias cue: matrilineal kin indicator per technique (N×K)
    "pr": stan_data["press"],  # Rank-bias cue: alpha rank indicator per technique (N×K)
    "co": stan_data["cohos"],  # Cohort-bias cue: age-similarity per technique (N×K)
    "yo": stan_data[
        "yobs"
    ],  # Age-bias (YOB) cue: year-of-birth prestige per technique (N×K)
    "age": stan_data["age"],  # Centered age of the monkey at each observation
}

m.fit(model=bi_ewa_model, obs=bi_data, num_warmup=1000, num_samples=1000, num_chains=2)
bi_summary_df = m.summary()
print("BI summary:\n", bi_summary_df)

# 4. Compare and log
bi_means = {"lambda": bi_summary_df.loc["lambda", "mean"]}
for i in range(8):
    bi_means[f"mu[{i+1}]"] = bi_summary_df.loc[f"mu[{i}]", "mean"]
    bi_means[f"sigma[{i+1}]"] = bi_summary_df.loc[f"sigma[{i}]", "mean"]
for i in range(2):
    bi_means[f"b_age[{i+1}]"] = bi_summary_df.loc[f"b_age[{i}]", "mean"]

log_lines = ["Parameter\tStan Mean\tBI Mean\tDifference"]
for p in stan_params:
    s_m = stan_means[p]
    b_m = bi_means.get(p, float("nan"))
    diff = b_m - s_m
    log_lines.append(f"{p}\t{s_m:.4f}\t{b_m:.4f}\t{diff:.4f}")

with open("log.txt", "w") as f:
    f.write("\n".join(log_lines))
print("\n".join(log_lines))

# 5. Density plots
samples_stan = fit_stan.draws_pd()
posteriors_bi = m.posteriors

ncols = 4
nrows = (len(stan_params) + ncols - 1) // ncols
plt.figure(figsize=(ncols * 4, nrows * 3))

for i, p in enumerate(stan_params):
    plt.subplot(nrows, ncols, i + 1)
    sns.kdeplot(samples_stan[p], label="Stan", color="blue")

    if "mu[" in p:
        bi_p = posteriors_bi["mu"][:, int(p[3:-1]) - 1]
    elif "sigma[" in p:
        bi_p = posteriors_bi["sigma"][:, int(p[6:-1]) - 1]
    elif "b_age[" in p:
        bi_p = posteriors_bi["b_age"][:, int(p[6:-1]) - 1]
    else:
        bi_p = posteriors_bi[p]

    sns.kdeplot(bi_p, label="BI", color="orange")
    plt.title(p)
    plt.legend(fontsize=7)

plt.tight_layout()
plt.savefig("density_comparison.png", dpi=120)
print("Done. Saved log.txt and density_comparison.png")

# %%
