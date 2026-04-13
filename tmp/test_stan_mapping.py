import numpy as np
import pandas as pd

# Mock samples from Stan (CmdStanModel returns numpy arrays)
stan_samples = {
    'block_effects': np.random.randn(100, 10),
    'sr_sigma': np.random.randn(100, 2),
    'dr_sigma': np.random.randn(100),
    'focal_effects': np.random.randn(100, 4),
    'target_effects': np.random.randn(100, 4),
    'dyad_effects': np.random.randn(100, 3)
}

stan_mapped = {}

# The mapping logic from SRM.py
try:
    stan_mapped['block_parameters_1'] = stan_samples['block_effects'][:, 0:1]
    stan_mapped['block_parameters_2'] = stan_samples['block_effects'][:, 1:10]
    stan_mapped['focal_effects'] = stan_samples['focal_effects']
    stan_mapped['target_effects'] = stan_samples['target_effects']
    stan_mapped['dyadic_coeffs'] = stan_samples['dyad_effects']
    stan_mapped['focal_target_sd'] = stan_samples['sr_sigma']
    stan_mapped['dyadic_sd'] = stan_samples['dr_sigma'].reshape(-1, 1)
    
    print("Mapping successful!")
    for k, v in stan_mapped.items():
        print(f"{k}: shape {v.shape}")
except Exception as e:
    print(f"Mapping failed: {e}")

# Verify extract_stats as well (from SRM.py)
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
                var_suffix = f"{i+1}_{j+1}"
                res.append({'mean': np.mean(data[:, i, j]), 'std': np.std(data[:, i, j]), 'var': f"{name}_{var_suffix}", 'Backend': backend})
        return pd.DataFrame(res)
    return pd.DataFrame()

all_res = []
for k in stan_mapped.keys():
    stats = extract_stats(stan_mapped, k, 'STAN')
    all_res.append(stats)

df = pd.concat(all_res, ignore_index=True)
print("\nStats extracted:")
print(df.head())
print(f"Total parameters: {len(df)}")
