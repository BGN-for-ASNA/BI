import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
from scipy.spatial.distance import jensenshannon

def calculate_kl_divergence(p_samples, q_samples, bins=100):
    """
    Calculate KL divergence between two distributions given their samples.
    p_samples: Ground truth (Stan)
    q_samples: Approximation (BI)
    """
    # Combine samples to find the range
    combined = np.concatenate([p_samples, q_samples])
    vmin, vmax = np.min(combined), np.max(combined)
    
    # Create histograms
    p_hist, _ = np.histogram(p_samples, bins=bins, range=(vmin, vmax), density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, range=(vmin, vmax), density=True)
    
    # Add small constant to avoid division by zero
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    
    # Normalize
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    
    return entropy(p_hist, q_hist)

def prepare_bi_data(m):
    #data_dict = m.sampler.get_samples()
    data_dict = m.posteriors
    # Initialize an empty DataFrame to collect parameters
    all_params = []

    # Loop through each array in the dictionary
    for key, array in data_dict.items():
        # Check the shape of the array
        if array.ndim > 1 and array.ndim < 3:
            # Create a DataFrame from the array and add a column for each parameter
            param_df = pd.DataFrame(array)
            # Rename columns to include the parameter name with standard indexing [0], [1], etc.
            param_df.columns = [f"{key}[{j}]" for j in range(array.shape[1])]
            all_params.append(param_df)
    
        elif array.ndim >= 3:# we have a matrix
            array_shape = array.shape
            # Assuming shape is (num_samples, dim1, dim2)
            dim1 = array_shape[1]
            dim2 = array_shape[2]
            for a in range(dim1):
                for b in range(dim2):
                    all_params.append(pd.DataFrame({f"{key}[{a}][{b}]": array[:,a,b]}))
        else:
            # If it's a 1D array, create a single column DataFrame
            all_params.append(pd.DataFrame({key: array}))

    # Concatenate all parameter DataFrames along the rows
    df_bi = pd.concat(all_params, axis=1)
    return df_bi

def prepare_stan_data(df):
    columns_to_remove = ['lp__', 'accept_stat__', 'stepsize__', 'treedepth__', 'n_leapfrog__', 'divergent__', 'energy__', 'chain__', 'iter__', 'draw__']
    d = df.drop(columns=[c for c in columns_to_remove if c in df.columns])
    return d

def build_stan_model(stan_code, data, chains=1, iter_sampling=1000, iter_warmup=1000):
    from cmdstanpy import CmdStanModel
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        stan_file = os.path.join(tmpdir, 'model.stan')
        with open(stan_file, 'w') as f:
            f.write(stan_code)
        model = CmdStanModel(stan_file=stan_file)
        fit = model.sample(data=data, chains=chains, iter_sampling=iter_sampling, iter_warmup=iter_warmup, show_progress=False)
        return fit.draws_pd()

def combine_data(df_bi, d):
    # Match columns by name where possible
    common_cols = list(set(df_bi.columns) & set(d.columns))
    if common_cols:
        df_bi_sub = df_bi[common_cols].copy()
        d_sub = d[common_cols].copy()
    else:
        # Fallback to positional if no names match (risky, but keeps existing behavior for mapped names)
        # However, if names were already mapped in plot_comparaison, they will match.
        d_sub = d.copy()
        df_bi_sub = df_bi.copy()
        if len(d_sub.columns) == len(df_bi_sub.columns):
            d_sub.columns = df_bi_sub.columns
            common_cols = df_bi_sub.columns.tolist()
        else:
            print("Warning: Column count mismatch in combine_data")
            return pd.concat([df_bi, d], ignore_index=True)

    df_bi_sub['method'] = 'BI'
    d_sub['method'] = 'STAN'
    d_comb = pd.concat([d_sub, df_bi_sub], ignore_index=True)
    return d_comb

def plot_comparaison(m, df, param_map=None, model_name=None):
    if isinstance(m, pd.DataFrame):
        df_bi = m.copy()
    else:
        df_bi = prepare_bi_data(m)
        
    d = prepare_stan_data(df).copy()
    
    if param_map:
        d = d[list(param_map.values())]
        d.columns = list(param_map.keys())
        df_bi = df_bi[list(param_map.keys())]
        
    d_comb = combine_data(df_bi, d)
    d_stan_clean = d_comb[d_comb['method'] == 'STAN']
    df_bi_clean = d_comb[d_comb['method'] == 'BI']

    params = [c for c in df_bi_clean.columns if c != 'method']
    num_params = len(params)
    num_cols = 3
    num_rows = (num_params + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, max(5, num_rows * 4)), sharey=False)
    if num_rows * num_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for a, i in enumerate(params):
        sns.kdeplot(data=d_comb, x=i, hue='method', ax=axes[a], fill=True, alpha=0.5)
        
        # Calculate KL Divergence
        p_samples = d_stan_clean[i].values
        q_samples = df_bi_clean[i].values
        kl_div = calculate_kl_divergence(p_samples, q_samples)
        
        axes[a].set_title(f"{i}\nKL Div: {kl_div:.4f}")
        if a % num_cols != 0:
            axes[a].set_ylabel('')

    plt.tight_layout()
    
    if model_name:
        import os
        plot_dir = "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Generate and save comparison table
        generate_comparison_table(df_bi_clean, d_stan_clean, model_name)
        
        plt.savefig(f"{plot_dir}/{model_name}_comparison.png")
        print(f"Plot saved to {plot_dir}/{model_name}_comparison.png")
        
    plt.close()
    return plt

def generate_comparison_table(df_bi, d, model_name):
    params = [c for c in df_bi.columns if c != 'method']
    table_data = []
    
    for p in params:
        bi_mean = df_bi[p].mean()
        stan_mean = d[p].mean()
        diff = bi_mean - stan_mean
        kl_div = calculate_kl_divergence(d[p].values, df_bi[p].values)
        
        table_data.append({
            'Parameter': p,
            'BI Mean': f"{bi_mean:.4f}",
            'Stan Mean': f"{stan_mean:.4f}",
            'Diff': f"{diff:.4f}",
            'KL Div': f"{kl_div:.4f}"
        })
    
    table_df = pd.DataFrame(table_data)
    
    # Save to txt file
    output_path = f"plots/{model_name}_comparison_table.txt"
    with open(output_path, 'w') as f:
        f.write(f"Comparison Table for {model_name}\n")
        f.write("="*60 + "\n")
        f.write(table_df.to_string(index=False))
        f.write("\n" + "="*60 + "\n")
    
    print(f"Comparison table saved to {output_path}")
    return table_df

def plot_recovery(results_df, model_name=None, r2_threshold=0.8):
    num_params = len(results_df['parameter'].unique())
    num_cols = 3
    num_rows = (num_params + num_cols - 1) // num_cols

    # Ensure numeric types for plotting
    results_df['simulated'] = pd.to_numeric(results_df['simulated'], errors='coerce')
    results_df['estimations'] = pd.to_numeric(results_df['estimations'], errors='coerce')

    g = sns.FacetGrid(results_df, col="parameter", col_wrap=num_cols, height=4, sharey=False, sharex=False)
    g.map(sns.scatterplot, "simulated", "estimations")
    
    low_r2_params = []
    
    for ax in g.axes.flat:
        low, high = ax.get_xlim()
        ax.plot([low, high], [low, high], color='red', ls='--')
        title = ax.get_title()
        param = title.split(" = ")[-1] if " = " in title else title
        sub = results_df[results_df['parameter'] == param].dropna(subset=['simulated', 'estimations'])
        if len(sub) >= 2:
            r2 = np.corrcoef(sub['simulated'], sub['estimations'])[0, 1] ** 2
            ax.set_title(f"{title}\n$R^2={r2:.2f}$")
            
            if r2 < r2_threshold:
                low_r2_params.append((param, r2))
    
    if model_name:
        import os
        plot_dir = "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # R2 report generation removed as per user request
        # report_path = f"{plot_dir}/{model_name}_r2_report.txt"
        # with open(report_path, "w") as f:
        #     f.write(f"R-squared Report for {model_name}\n")
        #     f.write("="*40 + "\n")
        #     if low_r2_params:
        #         f.write(f"WARNING: The following parameters have R^2 below {r2_threshold}:\n")
        #         for p, val in low_r2_params:
        #             f.write(f"- {p}: {val:.4f}\n")
        #     else:
        #         f.write(f"All parameters have R^2 above {r2_threshold}.\n")
        
        # print(f"R2 report saved to {report_path}")
        
        g.savefig(f"{plot_dir}/{model_name}_recovery.png")
        print(f"Plot saved to {plot_dir}/{model_name}_recovery.png")
        
    plt.close(g.fig)
    return g

