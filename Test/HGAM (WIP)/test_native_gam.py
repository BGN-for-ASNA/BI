import jax.numpy as jnp
import numpy as np
import pandas as pd
from gam_utils import gam
import os

def test_model_g_parity():
    print("\n--- Testing Model G Basis Parity ---")
    data_dir = "bi_data/G"
    bird_move = pd.read_csv("data/bird_move.csv")
    week, lat = bird_move['week'].values, bird_move['latitude'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_bi, S_bi = gam.hgam([week, lat], [5, 5], ["cc", "tp"], None, type="G")
    
    print(f"R Shape: {X_r.shape}")
    print(f"BI Shape: {X_bi.shape}")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_bi, X_r)
    recon_error = np.linalg.norm(X_r - X_bi @ beta_map) / np.linalg.norm(X_r)
    print(f"G Recon Error: {recon_error:.4f}")
    if recon_error < 0.15: print("SUCCESS")

def test_model_gs_parity():
    print("\n--- Testing Model GS Basis Parity ---")
    data_dir = "bi_data/GS"
    bird_move = pd.read_csv("data/bird_move.csv")
    week, lat, species = bird_move['week'].values, bird_move['latitude'].values, bird_move['species'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_bi, S_bi = gam.hgam([week, lat], [5, 5], ["cc", "tp"], species, type="GS")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_bi, X_r)
    results_r = pd.read_csv(f"{data_dir}/results_r.csv")
    beta_r = results_r['coef'].values
    pred_r = X_r @ beta_r
    pred_bi = X_bi @ (beta_map @ beta_r)
    pred_error = np.linalg.norm(pred_r - pred_bi) / np.linalg.norm(pred_r)
    
    print(f"R Shape: {X_r.shape}")
    print(f"BI Shape: {X_bi.shape}")
    print(f"Prediction Parity Error: {pred_error:.4f}")
    if pred_error < 0.1: print("SUCCESS")

def test_model_gi_parity():
    print("\n--- Testing Model GI Basis Parity ---")
    data_dir = "bi_data/GI"
    bird_move = pd.read_csv("data/bird_move.csv")
    week, lat, species = bird_move['week'].values, bird_move['latitude'].values, bird_move['species'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_bi, S_bi = gam.hgam([week, lat], [5, 5], ["cc", "tp"], species, type="GI")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_bi, X_r)
    results_r = pd.read_csv(f"{data_dir}/results_r.csv")
    beta_r = results_r['coef'].values
    pred_r = X_r @ beta_r
    pred_bi = X_bi @ (beta_map @ beta_r)
    pred_error = np.linalg.norm(pred_r - pred_bi) / np.linalg.norm(pred_r)
    
    print(f"R Shape: {X_r.shape}")
    print(f"BI Shape: {X_bi.shape}")
    print(f"Prediction Parity Error: {pred_error:.4f}")
    if pred_error < 0.1: print("SUCCESS")

def test_model_s_parity():
    print("\n--- Testing Model S Basis Parity ---")
    data_dir = "bi_data/S"
    bird_move = pd.read_csv("data/bird_move.csv")
    week, lat, species = bird_move['week'].values, bird_move['latitude'].values, bird_move['species'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_bi, S_bi = gam.hgam([week, lat], [5, 5], ["cc", "tp"], species, type="S")
    
    print(f"R Shape: {X_r.shape}")
    print(f"BI Shape: {X_bi.shape}")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_bi, X_r)
    recon_error = np.linalg.norm(X_r - X_bi @ beta_map) / np.linalg.norm(X_r)
    print(f"S Recon Error: {recon_error:.4f}")
    if recon_error < 0.2: print("SUCCESS")

def test_model_i_parity():
    print("\n--- Testing Model I Basis Parity ---")
    data_dir = "bi_data/I"
    bird_move = pd.read_csv("data/bird_move.csv")
    week, lat, species = bird_move['week'].values, bird_move['latitude'].values, bird_move['species'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_bi, S_bi = gam.hgam([week, lat], [5, 5], ["cc", "tp"], species, type="I")
    
    print(f"R Shape: {X_r.shape}")
    print(f"BI Shape: {X_bi.shape}")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_bi, X_r)
    recon_error = np.linalg.norm(X_r - X_bi @ beta_map) / np.linalg.norm(X_r)
    print(f"I Recon Error: {recon_error:.4f}")
    if recon_error < 0.2: print("SUCCESS")

if __name__ == "__main__":
    test_model_g_parity()
    test_model_gs_parity()
    test_model_gi_parity()
    test_model_s_parity()
    test_model_i_parity()
