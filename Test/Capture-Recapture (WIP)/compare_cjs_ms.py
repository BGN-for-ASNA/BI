import json
import jax.numpy as jnp
from model_BF import BF_mscr_model
from prepare_data import load_fleayi_data
from BayesForge import BayesForge
import numpyro
import time

def compare():
    # Load Stan results
    with open('stan_results.json', 'r') as f:
        stan_results = json.load(f)
    
    n_ind = 400
    data = load_fleayi_data(subset_n=n_ind)
    
    m = BF(platform='cpu')
    
    # Initialize with Stan means to speed up or check local neighborhood
    init_values = {
        'h': jnp.array(stan_results['h']),
        'q': jnp.array(stan_results['q']),
        'p_detect': jnp.array(stan_results['p']).reshape((3, 20))
    }
    
    print("Running BF model...")
    start_time = time.time()
    m.fit(
        model=BF_mscr_model,
        obs=data,
        num_warmup=150,
        num_samples=150,
        num_chains=1,
        seed=42,
        init_strategy=numpyro.infer.init_to_value(values=init_values)
    )
    BF_time = time.time() - start_time
    
    # Extract BF means
    posteriors = m.posteriors
    BF_h = jnp.mean(posteriors['h'], axis=0)
    BF_q = jnp.mean(posteriors['q'], axis=0)
    # p_detect is (S, Jm1). Stan flattens column-majorly.
    BF_p = jnp.mean(posteriors['p_detect'], axis=0).flatten(order='F')
    
    stan_h = stan_results['h']
    stan_q = stan_results['q']
    stan_p = stan_results['p']
    
    # Create log.txt
    with open('log.txt', 'w') as f:
        f.write(f"{'Parameter':<15} | {'Stan Mean':<12} | {'BF Mean':<12} | {'Diff':<12}\n")
        f.write("-" * 60 + "\n")
        
        # h parameters
        for i, (s, b) in enumerate(zip(stan_h, BF_h)):
            f.write(f"{'h['+str(i+1)+']':<15} | {s:12.4f} | {b:12.4f} | {b-s:12.4f}\n")
            
        # q parameters
        for i, (s, b) in enumerate(zip(stan_q, BF_q)):
            f.write(f"{'q['+str(i+1)+']':<15} | {s:12.4f} | {b:12.4f} | {b-s:12.4f}\n")
            
        # p parameters (first few for brevity in log, but all calculated)
        for i, (s, b) in enumerate(zip(stan_p[:5], BF_p[:5])):
            f.write(f"{'p['+str(i+1)+']':<15} | {s:12.4f} | {b:12.4f} | {b-s:12.4f}\n")
        f.write("... (truncated p list)\n")

    print("Comparison complete. Results saved to log.txt")
    
    # Save posterior samples for plotting
    jnp.savez('BF_samples.npz', h=posteriors['h'], q=posteriors['q'], p_detect=posteriors['p_detect'])

if __name__ == "__main__":
    compare()
