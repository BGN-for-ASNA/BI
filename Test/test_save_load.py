import numpy as np
import jax.numpy as jnp
from BI.Main.main import bi
import os

def test_save_load():
    # 1. Setup simple model and data
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1 + np.random.normal(0, 1, 20)
    
    def model(x, y=None):
        beta = bi_inst.dist.normal(0, 10, name="beta")
        alpha = bi_inst.dist.normal(0, 10, name="alpha")
        sigma = bi_inst.dist.half_normal(1, name="sigma")
        mu = alpha + beta * x
        bi_inst.dist.normal(mu, sigma, obs=y, name="Y")

    # 2. Fit the model
    bi_inst = bi(platform="cpu")
    print("Fitting model...")
    bi_inst.fit(model=model, obs={'x': x, 'y': y}, num_samples=100, num_warmup=100, num_chains=1)
    
    # Store some attributes for comparison
    original_posteriors = bi_inst.posteriors
    original_data = bi_inst.data_on_model
    original_model_name = bi_inst.model_name
    
    save_path = "test_bi_model.pkl"
    
    # 3. Save the object without a path
    print("Saving model without path...")
    bi_inst.save()
    
    # Expected filename based on model name
    save_path = f"{bi_inst.model_name}_bi.pkl"
    
    # 4. Load the object without a path
    print(f"Loading model from {save_path}...")
    loaded_bi = bi.load(save_path)
    
    # 5. Verify
    print("\nVerification:")
    
    # Check data
    data_match = all(np.allclose(original_data[k], loaded_bi.data_on_model[k]) for k in original_data)
    print(f"Data matches: {data_match}")
    
    # Check posteriors
    post_match = all(np.allclose(original_posteriors[k], loaded_bi.posteriors[k]) for k in original_posteriors)
    print(f"Posteriors match: {post_match}")
    
    # Check model name
    print(f"Model name matches: {original_model_name == loaded_bi.model_name}")
    
    # Check if methods still work
    try:
        summ = loaded_bi.summary()
        print("Summary method works on loaded object.")
        # print(summ)
    except Exception as e:
        print(f"Summary method failed: {e}")

    # Cleanup (removing the picked file but keeping this test script)
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_save_load()
