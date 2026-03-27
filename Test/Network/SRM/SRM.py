import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from BI import bi, jnp
import os
from cmdstanpy import CmdStanModel
import sys

# Setup device and BI model ------------------------------------------------
m = bi(platform='cpu', rand_seed = False)

# 1. Simulate data ------------------------------------------------
print("Step 1: Simulating data...")
sys.stdout.flush()
N = 50 
N_focal_vars = 4
N_target_vars = 4
N_dyad_vars = 3

focal_individual_predictors = np.random.normal(0, 1, size=(N, N_focal_vars))
target_individual_predictors = np.random.normal(0, 1, size=(N, N_target_vars))

dyadic_predictors_mat = np.random.binomial(1, 0.3, size=(N, N, N_dyad_vars))
for k in range(N_dyad_vars):
    np.fill_diagonal(dyadic_predictors_mat[:, :, k], 0)

dyadic_predictors_edgl = jnp.stack([m.net.mat_to_edgl(dyadic_predictors_mat[:, :, k]) for k in range(N_dyad_vars)], axis=2)

category = np.random.choice([0, 1, 2], size=(N,))
N_grp = 3
N_by_grp = np.array([np.sum(category == i) for i in range(N_grp)])

def sim_network(dyadic_predictors, focal_individual_predictors, target_individual_predictors, category):
    N_id = focal_individual_predictors.shape[0]
    B_intercept = m.net.block_model(jnp.full((N_id,), 0), 1, jnp.ones(1)*N_id, sample=True, name="intercept")
    B_category = m.net.block_model(category, N_grp, jnp.array(N_by_grp), sample=True, name="category")
    sr = m.net.sender_receiver(
        jnp.array(focal_individual_predictors), 
        jnp.array(target_individual_predictors), 
        s_mu=0.4, r_mu=-0.4, sample=True)
    dr = m.net.dyadic_effect(dyadic_predictors, d_sd=2.5, sample=True)
    logits = B_intercept + B_category + sr + dr
    return m.dist.bernoulli(logits=logits, sample=True)

network_edgl = sim_network(dyadic_predictors_edgl, focal_individual_predictors, target_individual_predictors, category)
print(f"Network simulated. Edgelist shape: {network_edgl.shape}")
sys.stdout.flush()

# 2. Fit BI Model ------------------------------------------------
print("Step 2: Fitting BI model...")
sys.stdout.flush()
m.data_on_model = dict(
    network=network_edgl, 
    dyadic_predictors=dyadic_predictors_edgl,
    focal_individual_predictors=jnp.array(focal_individual_predictors),
    target_individual_predictors=jnp.array(target_individual_predictors), 
    category=jnp.array(category)
)

def model(network, dyadic_predictors, focal_individual_predictors, target_individual_predictors, category):
    N_id = focal_individual_predictors.shape[0]
    B_intercept = m.net.block_model(jnp.full((N_id,), 0), 1, jnp.ones(1)*N_id, name="B_intercept")
    B_category = m.net.block_model(category, N_grp, jnp.array(N_by_grp), name="B_category")
    sr = m.net.sender_receiver(focal_individual_predictors, target_individual_predictors, s_mu=0.1, r_mu=0.01)
    dr = m.net.dyadic_effect(dyadic_predictors, d_sd=2.5) 
    m.dist.bernoulli(logits=B_intercept + B_category + sr + dr, obs=network)

try:
    m.fit(model, num_samples=500, num_warmup=500, progress_bar=True)
    bi_samples = m.posteriors 
    print("BI model fitting complete.")
    sys.stdout.flush()
except Exception as e:
    print(f"BI model fitting failed: {e}")
    sys.stdout.flush()
    sys.exit(1)

# 3. Fit Stan Model ------------------------------------------------
print("Step 3: Preparing data for Stan...")
sys.stdout.flush()

try:
    network_matrix = np.zeros((N, N))
    urows, ucols = np.triu_indices(N, k=1)
    network_matrix[urows, ucols] = network_edgl[:, 0]
    network_matrix[ucols, urows] = network_edgl[:, 1]

    focal_set = np.column_stack([np.ones(N), focal_individual_predictors])
    target_set = np.column_stack([np.ones(N), target_individual_predictors])
    dyad_set = np.concatenate([np.ones((N, N, 1)), dyadic_predictors_mat], axis=2)
    block_set = np.column_stack([np.ones(N, dtype=int), category + 1])

    # Exact provided priors
    priors = np.zeros((23, 2))
    p_data = [
        [-3.00, 1.5], [3.00, 1.5], [-1.50, 1.0], [1.00, 0.0], [1.00, 0.0], [1.00, 0.0],
        [0.00, 2.5], [0.00, 2.5], [0.00, 2.5], [0.10, 2.5], [0.01, 2.5], [0.00, 2.5],
        [0.00, 2.5], [0.00, 2.5], [0.00, 2.5], [0.00, 2.5], [2.50, 0.0], [2.50, 0.0],
        [1.50, 0.0], [3.00, 1.0], [2.00, 0.0], [3.00, 12.0], [0.00, 2.5]
    ]
    for i, row in enumerate(p_data):
        priors[i] = row

    stan_data = {
        'N_id': N,
        'N_responses': 1,
        'N_params': [N_focal_vars + 1, N_target_vars + 1, N_dyad_vars + 1],
        'N_group_vars': 2,
        'max_N_groups': N_grp,
        'N_groups_per_var': [1, N_grp],
        'block_set': block_set.tolist(),
        'focal_set': focal_set.tolist(),
        'target_set': target_set.tolist(),
        'dyad_set': dyad_set.tolist(),
        'outcomes': network_matrix.astype(int).reshape(N, N, 1).tolist(),
        'outcomes_real': network_matrix.astype(float).reshape(N, N, 1).tolist(),
        'exposure': np.ones((N, N, 1), dtype=int).tolist(),
        'mask': np.zeros((N, N, 1), dtype=int).tolist(),
        'priors': priors.tolist(),
        'export_network': 0,
        'outcome_mode': 1,
        'link_mode': 1
    }

    print("Stan data preparation complete.")
    sys.stdout.flush()

    # SRM2 (edgelist) data preparation
    urows, ucols = np.triu_indices(N, k=1)
    N_obs = N * (N - 1)
    sender = np.concatenate([urows, ucols]) + 1
    receiver = np.concatenate([ucols, urows]) + 1
    dyad_id = np.concatenate([np.arange(1, len(urows) + 1), np.arange(1, len(urows) + 1)])
    dyad_dir = np.concatenate([np.ones(len(urows), dtype=int), np.full(len(urows), 2, dtype=int)])
    
    # Outcomes for SRM2 (long format)
    outcomes_srm2 = network_edgl.T.flatten() # [batch_i<j, batch_j<i]
    
    # Dyadic predictors for SRM2 (long format)
    # dyadic_predictors_edgl is (1225, 2, 3). We need (2450, 3) 
    # Plus the intercept column (column 0 in dyad_set)
    dyad_preds_long = np.concatenate([dyadic_predictors_edgl[:, 0, :], dyadic_predictors_edgl[:, 1, :]], axis=0)
    dyad_set_srm2 = np.column_stack([np.ones(N_obs), dyad_preds_long])

    stan_data_stan2 = {
        'N_networktypes': 1,
        'N_id': N,
        'N_dyads': len(urows),
        'N_obs': N_obs,
        'N_responses': 1,
        'N_params': [N_focal_vars + 1, N_target_vars + 1, N_dyad_vars + 1],
        'sender': sender.tolist(),
        'receiver': receiver.tolist(),
        'dyad_id': dyad_id.tolist(),
        'dyad_dir': dyad_dir.tolist(),
        'outcomes': outcomes_srm2.astype(int).tolist(),
        'outcomes_real': outcomes_srm2.astype(float).tolist(),
        'exposure': np.ones(N_obs, dtype=int).tolist(),
        'N_group_vars': 2,
        'max_N_groups': N_grp,
        'N_groups_per_var': [1, N_grp],
        'block_set': block_set.tolist(),
        'focal_set': focal_set.tolist(),
        'target_set': target_set.tolist(),
        'dyad_set': dyad_set_srm2.tolist(),
        'priors': priors.tolist(),
        'export_network': 0,
        'outcome_mode': 1,
        'link_mode': 1
    }

except Exception as e:
    print(f"Stan data preparation failed: {e}")
    sys.stdout.flush()
    sys.exit(1)

print("Fitting Stan models...")
sys.stdout.flush()
try:
    stan_file = r'C:\Users\Sosa\Documents\BI\Test\Network\SRM\SRM.stan'
    exe_file = r'C:\Users\Sosa\Documents\BI\Test\Network\SRM\SRM.exe'
    
    if os.path.exists(exe_file):
         sm = CmdStanModel(stan_file=stan_file, exe_file=exe_file)
    else:
         sm = CmdStanModel(stan_file=stan_file)
    
    fit = sm.sample(data=stan_data, iter_sampling=500, iter_warmup=500, chains=1, show_progress=False)
    stan_samples = fit.stan_variables()
    print("Stan model (SRM) fitting complete.")
    sys.stdout.flush()

    print("Fitting STAN2 (optimized edgelist with blocks) model...")
    stan_file2 = r'C:\Users\Sosa\Documents\BI\Test\Network\SRM\STAN2.stan'
    exe_file2 = r'C:\Users\Sosa\Documents\BI\Test\Network\SRM\STAN2.exe'
    if os.path.exists(exe_file2):
         sm2 = CmdStanModel(stan_file=stan_file2, exe_file=exe_file2)
    else:
         sm2 = CmdStanModel(stan_file=stan_file2)
    
    fit2 = sm2.sample(data=stan_data_stan2, iter_sampling=500, iter_warmup=500, chains=1, show_progress=False)
    stan2_samples = fit2.stan_variables()
    print("Stan model (STAN2) fitting complete.")
    sys.stdout.flush()
except Exception as e:
    print(f"Stan model fitting failed: {e}")
    sys.stdout.flush()
    sys.exit(1)

# 4. Posterior Comparison and Plotting ----------------------------------------
print("Step 4: Generating comparison plot...")
sys.stdout.flush()

def extract_stats(samples, name, backend):
    data = samples[name]
    if isinstance(data, list): data = np.array(data)
    if data.ndim == 1:
        return pd.DataFrame({'mean': [np.mean(data)], 'std': [np.std(data)], 'var': [name], 'Backend': [backend]})
    elif data.ndim == 2:
        res = []
        for i in range(data.shape[1]):
            res.append({'mean': np.mean(data[:, i]), 'std': np.std(data[:, i]), 'var': f"{name}_{i+1}", 'Backend': backend})
        return pd.DataFrame(res)
    elif data.ndim == 3:
        res = []
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                res.append({'mean': np.mean(data[:, i, j]), 'std': np.std(data[:, i, j]), 'var': f"{name}_{i+1}_{j+1}", 'Backend': backend})
        return pd.DataFrame(res)
    return pd.DataFrame()

try:
    bi_mapped = {}
    bi_mapped['block_parameters_1'] = bi_samples['b_B_intercept'].reshape(bi_samples['b_B_intercept'].shape[0], -1) 
    # Use Fortran order for BI flattening to match Stan's column-major to_matrix
    bi_mapped['block_parameters_2'] = np.array([m.flatten(order='F') for m in bi_samples['b_B_category']])
    bi_mapped['focal_effects'] = bi_samples['sender_effects']
    bi_mapped['target_effects'] = bi_samples['receiver_effects']
    bi_mapped['dyadic_coeffs'] = bi_samples['dyad_effects']
    bi_mapped['focal_target_sd'] = bi_samples['sr_sigma']
    bi_mapped['dyadic_sd'] = bi_samples['dr_sigma']

    stan_mapped = {}
    stan_mapped['block_parameters_1'] = stan_samples['block_effects'][:, 0:1]
    stan_mapped['block_parameters_2'] = stan_samples['block_effects'][:, 1:10]
    stan_mapped['focal_effects'] = stan_samples['focal_effects']
    stan_mapped['target_effects'] = stan_samples['target_effects']
    stan_mapped['dyadic_coeffs'] = stan_samples['dyad_effects']
    stan_mapped['focal_target_sd'] = stan_samples['sr_sigma']
    stan_mapped['dyadic_sd'] = stan_samples['dr_sigma'].reshape(-1, 1)

    stan2_mapped = {}
    stan2_mapped['block_parameters_1'] = stan2_samples['block_effects'][:, 0:1]
    stan2_mapped['block_parameters_2'] = stan2_samples['block_effects'][:, 1:10]
    stan2_mapped['focal_effects'] = stan2_samples['focal_effects']
    stan2_mapped['target_effects'] = stan2_samples['target_effects']
    stan2_mapped['dyadic_coeffs'] = stan2_samples['dyad_effects']
    stan2_mapped['focal_target_sd'] = stan2_samples['sr_sigma']
    stan2_mapped['dyadic_sd'] = stan2_samples['dr_sigma'].reshape(-1, 1)

    all_res = []
    for k in bi_mapped:
        all_res.append(extract_stats(bi_mapped, k, 'BI'))
        all_res.append(extract_stats(stan_mapped, k, 'STAN'))
        # Only add keys that exist in stan2_mapped
        if k in stan2_mapped:
            all_res.append(extract_stats(stan2_mapped, k, 'STAN2'))

    df = pd.concat(all_res)
    
    # Print comparison for all parameters to log
    print("\n--- All Parameter Means Comparison ---")
    pivot_df = df.pivot(index='var', columns='Backend', values='mean')
    print(pivot_df)
    sys.stdout.flush()

    # Column-Major labels (matches Stan's indexing)
    labels = {
        "block_parameters_1_1": "Block_intercept",
        "block_parameters_2_1": "Block_Merica 1-1",
        "block_parameters_2_2": "Block_Merica 2-1",
        "block_parameters_2_3": "Block_Merica 3-1",
        "block_parameters_2_4": "Block_Merica 1-2",
        "block_parameters_2_5": "Block_Merica 2-2",
        "block_parameters_2_6": "Block_Merica 3-2",
        "block_parameters_2_7": "Block_Merica 1-3",
        "block_parameters_2_8": "Block_Merica 2-3",
        "block_parameters_2_9": "Block_Merica 3-3",
        "target_effects_1": "Receiver_mass_Effect",
        "target_effects_2": "Receiver_age_Effect",
        "target_effects_3": "Receiver_love_Effect",
        "target_effects_4": "Receiver_fire_Effect",
        "focal_effects_1": "Sender_Mass_Effect",
        "focal_effects_2": "Sender_Age_Effect",
        "focal_effects_3": "Sender_love_Effect",
        "focal_effects_4": "Sender_fire_Effect",
        "focal_target_sd_1": "Nodes_random_Effect_sd1",
        "focal_target_sd_2": "Nodes_random_Effect_sd2",
        "dyadic_coeffs_1": "Dyads_Kinship_Effect",
        "dyadic_coeffs_2": "Dyads_Dominance_Effect",
        "dyadic_coeffs_3": "Dyads_Interaction_Effect",
        "dyadic_sd_1": "Dyads_Effect_sd"
    }

    df['label'] = df['var'].map(labels)
    df = df.dropna(subset=['label'])

    label_order = [
         "Receiver_fire_Effect", "Receiver_love_Effect", "Receiver_age_Effect", "Receiver_mass_Effect",
        "Nodes_random_Effect_sd2", "Nodes_random_Effect_sd1",
        "Sender_fire_Effect", "Sender_love_Effect", "Sender_Age_Effect", "Sender_Mass_Effect",
        "Dyads_Effect_sd", "Dyads_Interaction_Effect", "Dyads_Dominance_Effect", "Dyads_Kinship_Effect",
        "Block_Merica 3-3", "Block_Merica 2-3", "Block_Merica 1-3",
        "Block_Merica 3-2", "Block_Merica 2-2", "Block_Merica 1-2",
        "Block_Merica 3-1", "Block_Merica 2-1", "Block_Merica 1-1",
        "Block_intercept"
    ]
    df['label'] = pd.Categorical(df['label'], categories=label_order[::-1], ordered=True)
    df = df.sort_values('label')

    plt.figure(figsize=(12, 14))
    sns.set_style("whitegrid")
    palette = {'BI': '#F8766D', 'STAN': '#00BFC4', 'STAN2': '#7CAE00'}

    for i, backend in enumerate(['BI', 'STAN', 'STAN2']):
        subset = df[df['Backend'] == backend]
        y_indices = np.array([label_order[::-1].index(l) for l in subset['label']])
        if backend == 'STAN':
            offset = 0.2
        elif backend == 'STAN2':
            offset = 0.0
        else: # BI
            offset = -0.2
        
        plt.errorbar(subset['mean'], y_indices + offset, xerr=subset['std'], 
                     fmt='o', color=palette[backend], label=backend, capsize=0, markersize=8, elinewidth=3)

    plt.yticks(np.arange(len(label_order)), label_order[::-1], fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Posteriors", fontsize=16, fontweight='bold')
    plt.ylabel("SRM parameters", fontsize=16, fontweight='bold')
    plt.legend(title="Backend", loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=14)
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, 'srm_comparison.png')
    plt.savefig(plot_path)
    print(f"Comparison plot saved as {plot_path}")
    sys.stdout.flush()
except Exception as e:
    print(f"Plotting failed: {e}")
    sys.stdout.flush()
    sys.exit(1)