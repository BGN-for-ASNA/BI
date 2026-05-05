import sys
import re
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import cmdstanpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from BI import bi
from BI.Resources.datasets import load as bi_load

# Load BI/model.py as a module without making BI/ a package
_here = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("bi_model", _here / "BI_backend" / "model.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_model = _mod.make_model

# 1. Load data via BI loader
data_path = bi_load().panama_ewa(only_path=True)
d = pd.read_csv(data_path)

for col_prefix in ['Yobs', 'cos', 'ks', 'ps', 'prs']:
    cols = [f'{col_prefix}{i}' for i in range(1, 8)]
    max_val = d[cols].max().max()
    d[cols] = d[cols] / max_val

stan_data = {
    'K': 7,
    'N': len(d),
    'J': d['mono_index'].nunique(),
    'tech':  d['tech_index'].values.astype(int),
    'y':     d[[f'y{i}'    for i in range(1, 8)]].values,
    's':     d[[f's{i}'    for i in range(1, 8)]].values,
    'ps':    d[[f'ps{i}'   for i in range(1, 8)]].values,
    'ks':    d[[f'ks{i}'   for i in range(1, 8)]].values,
    'press': d[[f'prs{i}'  for i in range(1, 8)]].values,
    'cohos': d[[f'cos{i}'  for i in range(1, 8)]].values,
    'yobs':  d[[f'Yobs{i}' for i in range(1, 8)]].values,
    'bout':  d['forg_bout'].values.astype(int),
    'id':    d['mono_index'].values.astype(int),
    'N_effects': 8,
    'age':   d['age.c'].values,
}

# 2. Run Stan model
print("Running Stan model...")
stan_src  = 'Stan/PN_social_global_age.stan'
stan_tmp  = 'Stan/temp_PN_social_global_age.stan'

with open(stan_src) as f:
    content = f.read()

content = content.replace('#//', '//').replace('#', '//')
content = re.sub(r'generated quantities\s*\{.*\}', '', content, flags=re.DOTALL)
content = re.sub(r'(int|real)\s+([a-zA-Z0-9_]+)\[([^\]]+)\];', r'array[\3] \1 \2;', content)

with open(stan_tmp, 'w') as f:
    f.write(content)

sm       = cmdstanpy.CmdStanModel(stan_file=stan_tmp)
fit_stan = sm.sample(
    data=stan_data,
    iter_sampling=1000, iter_warmup=1000,
    chains=2, parallel_chains=2,
    adapt_delta=0.9,
)

stan_summary = fit_stan.summary()
stan_params  = [
    "lambda",
    "mu[1]", "mu[2]", "mu[3]", "mu[4]", "mu[5]", "mu[6]", "mu[7]", "mu[8]",
    "b_age[1]", "b_age[2]",
    "sigma[1]", "sigma[2]", "sigma[3]", "sigma[4]", "sigma[5]", "sigma[6]", "sigma[7]", "sigma[8]",
]
stan_means = {p: stan_summary.loc[p, 'Mean'] for p in stan_params}
print("Stan means:", stan_means)

# 3. Run BI model
print("Running BI model...")

y_prev      = np.zeros_like(stan_data['y'])
y_prev[1:]  = stan_data['y'][:-1]

m             = bi(platform='cpu')
bi_ewa_model  = make_model(m)

bi_data = {
    'K':    7,
    'J':    stan_data['J'],
    'tech': stan_data['tech'] - 1,
    'id':   stan_data['id']   - 1,
    'bout': stan_data['bout'],
    'y_prev': y_prev,
    's':    stan_data['s'],
    'ps':   stan_data['ps'],
    'ks':   stan_data['ks'],
    'pr':   stan_data['press'],
    'co':   stan_data['cohos'],
    'yo':   stan_data['yobs'],
    'age':  stan_data['age'],
}

m.fit(model=bi_ewa_model, obs=bi_data, num_warmup=1000, num_samples=1000, num_chains=2)
bi_summary_df = m.summary()
print("BI summary:\n", bi_summary_df)

# 4. Compare and log
bi_means = {"lambda": bi_summary_df.loc["lambda", "mean"]}
for i in range(8):
    bi_means[f"mu[{i+1}]"]    = bi_summary_df.loc[f"mu[{i}]",    "mean"]
    bi_means[f"sigma[{i+1}]"] = bi_summary_df.loc[f"sigma[{i}]", "mean"]
for i in range(2):
    bi_means[f"b_age[{i+1}]"] = bi_summary_df.loc[f"b_age[{i}]", "mean"]

log_lines = ["Parameter\tStan Mean\tBI Mean\tDifference"]
for p in stan_params:
    s_m  = stan_means[p]
    b_m  = bi_means.get(p, float('nan'))
    diff = b_m - s_m
    log_lines.append(f"{p}\t{s_m:.4f}\t{b_m:.4f}\t{diff:.4f}")

with open("log.txt", "w") as f:
    f.write("\n".join(log_lines))
print("\n".join(log_lines))

# 5. Density plots
samples_stan  = fit_stan.draws_pd()
posteriors_bi = m.posteriors

ncols = 4
nrows = (len(stan_params) + ncols - 1) // ncols
plt.figure(figsize=(ncols * 4, nrows * 3))

for i, p in enumerate(stan_params):
    plt.subplot(nrows, ncols, i + 1)
    sns.kdeplot(samples_stan[p], label='Stan', color='blue')

    if "mu[" in p:
        bi_p = posteriors_bi["mu"][:, int(p[3:-1]) - 1]
    elif "sigma[" in p:
        bi_p = posteriors_bi["sigma"][:, int(p[6:-1]) - 1]
    elif "b_age[" in p:
        bi_p = posteriors_bi["b_age"][:, int(p[6:-1]) - 1]
    else:
        bi_p = posteriors_bi[p]

    sns.kdeplot(bi_p, label='BI', color='orange')
    plt.title(p)
    plt.legend(fontsize=7)

plt.tight_layout()
plt.savefig("density_comparison.png", dpi=120)
print("Done. Saved log.txt and density_comparison.png")
