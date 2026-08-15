import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
from BayesForge import BayesForge
import json

def run_bi_benchmark(model_name):
    print(f"\n--- Running BF Model {model_name} ---")
    data_dir = f"BF_data/{model_name}"
    
    # Load data
    try:
        X = pd.read_csv(f"{data_dir}/X.csv").values
        y = pd.read_csv(f"{data_dir}/y.csv").values.flatten().astype(float)
        info = pd.read_csv(f"{data_dir}/smooth_info.csv")
        lambdas = pd.read_csv(f"{data_dir}/lambdas.csv")
    except FileNotFoundError:
        print(f"Data for {model_name} not found. Skipping.")
        return
        
    n_obs, n_params = X.shape
    
    # Initialize BF
    m = BF(platform='cpu')
    
    # Pre-calculate the combined precision matrix K
    K = np.zeros((n_params, n_params))
    smooth_params = []
    
    for idx, row in info.iterrows():
        start = int(row['first']) - 1
        end = int(row['last'])
        smooth_params.extend(range(start, end))
        
        # Add S matrices for this smooth
        prev_s_count = info.iloc[:idx]['n_S'].sum()
        for k in range(1, int(row['n_S']) + 1):
            s_file = f"{data_dir}/S_{row['id']}_{k}.csv"
            if os.path.exists(s_file):
                S_mat = pd.read_csv(s_file).values
                lambda_val = lambdas.iloc[int(prev_s_count + k - 1)]['lambda']
                K[start:end, start:end] += lambda_val * S_mat
            
    # Parametric parts get flat prior (tiny precision)
    for i in range(n_params):
        if i not in smooth_params:
            K[i, i] = 1e-4
            
    # Stabilization
    K += np.eye(n_params) * 1e-6
    cov = np.linalg.inv(K)

    # Set data on model
    obs_data = {'X_data': jnp.array(X), 'y_obs': jnp.array(y)}

    def model_func(**data):
        beta = m.dist.multivariate_normal(
            loc=jnp.zeros(n_params),
            covariance_matrix=jnp.array(cov),
            name='beta'
        )
        linear_predictor = jnp.dot(data['X_data'], beta)
        rate = jnp.exp(linear_predictor)
        m.dist.poisson(rate=rate, obs=data['y_obs'], name='obs')

    # Fit the model
    # Use 1 chain, 1000 samples for benchmarking speed
    m.fit(model_func, obs=obs_data, num_samples=1000, num_warmup=500, num_chains=1)
    
    # Get summary from posteriors
    post = m.posteriors['beta']
    # post is likely a JAX array or numpy array/DataArray
    # From Voice/BF_only.py it looks like an array we can index or take mean of
    beta_means = np.array(post).mean(axis=0)
    
    # Save results
    res_df = pd.DataFrame({'BF_mean': beta_means})
    res_df.to_csv(f"{data_dir}/results_BF.csv", index=False)
    print(f"BF Model {model_name} finished.")

if __name__ == "__main__":
    for model in ['G', 'GS', 'GI', 'S', 'I']:
        try:
            run_bi_benchmark(model)
        except Exception as e:
            print(f"Error in model {model}: {e}")
            import traceback
            traceback.print_exc()
