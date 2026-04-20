import json
import jax.numpy as jnp

def load_fleayi_data(subset_n=None):
    # Load from the JSON exported by R
    with open('data_bi.json', 'r') as f:
        data = json.load(f)
    
    y = jnp.array(data['y'])
    f = jnp.array(data['f'])
    l = jnp.array(data['l'])
    tau = jnp.array(data['tau'])
    
    # subset_n is already handled in export_data.R (fixed at 400 for consistency)
    # but we can subset further if needed
    if subset_n and subset_n < y.shape[0]:
        y = y[:subset_n]
        f = f[:subset_n]
        l = l[:subset_n]
        
    return {
        'y': y,
        'f': f,
        'l': l,
        'tau': tau,
        'n_individuals': y.shape[0],
        'n_surveys': y.shape[1],
        'n_states': 3
    }

if __name__ == "__main__":
    data = load_fleayi_data(subset_n=400)
    print(f"Data loaded: {data['n_individuals']} individuals, {data['n_surveys']} surveys")
