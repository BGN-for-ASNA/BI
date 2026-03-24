import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_bi_data(m):
    #data_dict = m.sampler.get_samples()
    data_dict = m.posteriors
    # Initialize an empty DataFrame to collect parameters
    all_params = []

    # Loop through each array in the dictionary
    for key, array in data_dict.items():
        # TFP and other backends might return (chains, draws, ...)
        # If we have a chain dimension, we flatten it for the comparison plots
        if array.ndim >= 2:
            # Check if this looks like (chains, draws, ...)
            # In BI, num_chains is stored in m.num_chains
            if hasattr(m, 'num_chains') and array.shape[0] == m.num_chains:
                # Flatten chains and draws into a single samples dimension
                new_shape = (-1,) + array.shape[2:]
                array = array.reshape(new_shape)

        # Now handle the flattened array
        if array.ndim == 1:
            all_params.append(pd.DataFrame({key: array}))
        elif array.ndim == 2:
            # (samples, params)
            param_df = pd.DataFrame(array)
            param_df.columns = [f"{key}_{j+1}" for j in range(array.shape[1])]
            all_params.append(param_df)
        elif array.ndim >= 3:
            # (samples, row, col, ...)
            # Flatten everything after the first dimension
            samples = array.shape[0]
            rest = array.shape[1:]
            import itertools
            indices = list(itertools.product(*[range(d) for d in rest]))
            for idx in indices:
                col_name = f"{key}_" + "_".join(map(str, idx))
                # Take all samples for this specific element
                indexer = (slice(None),) + idx
                all_params.append(pd.DataFrame({col_name: array[indexer]}))

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
    #df_bi = pd.DataFrame(samples)
    params = df_bi.columns.values
    d.columns = df_bi.columns

    df_bi['method'] = 'BI'
    d['method'] = 'STAN'
    d_comb = pd.concat([d, df_bi], ignore_index=True)
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

    params = df_bi.columns.values[:-1]
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
        axes[a].set_title(i)
        if a % num_cols != 0:
            axes[a].set_ylabel('')

    plt.tight_layout()
    
    if model_name:
        import os
        plot_dir = "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plt.savefig(f"{plot_dir}/{model_name}_comparison.png")
        print(f"Plot saved to {plot_dir}/{model_name}_comparison.png")
        
    plt.close()
    return plt

def plot_recovery(results_df, model_name=None):
    num_params = len(results_df['parameter'].unique())
    num_cols = 3
    num_rows = (num_params + num_cols - 1) // num_cols
    
    g = sns.FacetGrid(results_df, col="parameter", col_wrap=num_cols, height=4, sharey=False, sharex=False)
    g.map(sns.scatterplot, "simulated", "estimations")
    for ax in g.axes.flat:
        low, high = ax.get_xlim()
        ax.plot([low, high], [low, high], color='red', ls='--')
    
    if model_name:
        import os
        plot_dir = "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        g.savefig(f"{plot_dir}/{model_name}_recovery.png")
        print(f"Plot saved to {plot_dir}/{model_name}_recovery.png")
        
    plt.close(g.fig)
    return g

