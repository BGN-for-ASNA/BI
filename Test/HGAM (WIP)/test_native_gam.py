import os
import pathlib

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from gam_utils import gam

_HERE = pathlib.Path(__file__).resolve().parent
_DATA = _HERE / "data" / "bird_move.csv"

pytestmark = pytest.mark.skipif(
    not _DATA.exists(),
    reason=f"HGAM reference data not found at {_DATA}; "
           "generate it with the R scripts in this directory first.",
)

def test_model_g_parity():
    print("\n--- Testing Model G Basis Parity ---")
    data_dir = str(_HERE / "BF_data" / "G")
    bird_move = pd.read_csv(_DATA)
    week, lat = bird_move['week'].values, bird_move['latitude'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_BF, S_BF = gam.hgam([week, lat], [5, 5], ["cc", "tp"], None, type="G")
    
    print(f"R Shape: {X_r.shape}")
    print(f"BF Shape: {X_BF.shape}")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_BF, X_r)
    recon_error = np.linalg.norm(X_r - X_BF @ beta_map) / np.linalg.norm(X_r)
    print(f"G Recon Error: {recon_error:.4f}")
    assert recon_error < 0.15, f"recon_error={recon_error:.4f} exceeds tolerance 0.15"

def test_model_gs_parity():
    print("\n--- Testing Model GS Basis Parity ---")
    data_dir = str(_HERE / "BF_data" / "GS")
    bird_move = pd.read_csv(_DATA)
    week, lat, species = bird_move['week'].values, bird_move['latitude'].values, bird_move['species'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_BF, S_BF = gam.hgam([week, lat], [5, 5], ["cc", "tp"], species, type="GS")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_BF, X_r)
    results_r = pd.read_csv(f"{data_dir}/results_r.csv")
    beta_r = results_r['coef'].values
    pred_r = X_r @ beta_r
    pred_BF = X_BF @ (beta_map @ beta_r)
    pred_error = np.linalg.norm(pred_r - pred_BF) / np.linalg.norm(pred_r)
    
    print(f"R Shape: {X_r.shape}")
    print(f"BF Shape: {X_BF.shape}")
    print(f"Prediction Parity Error: {pred_error:.4f}")
    assert pred_error < 0.1, f"pred_error={pred_error:.4f} exceeds tolerance 0.1"

def test_model_gi_parity():
    print("\n--- Testing Model GI Basis Parity ---")
    data_dir = str(_HERE / "BF_data" / "GI")
    bird_move = pd.read_csv(_DATA)
    week, lat, species = bird_move['week'].values, bird_move['latitude'].values, bird_move['species'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_BF, S_BF = gam.hgam([week, lat], [5, 5], ["cc", "tp"], species, type="GI")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_BF, X_r)
    results_r = pd.read_csv(f"{data_dir}/results_r.csv")
    beta_r = results_r['coef'].values
    pred_r = X_r @ beta_r
    pred_BF = X_BF @ (beta_map @ beta_r)
    pred_error = np.linalg.norm(pred_r - pred_BF) / np.linalg.norm(pred_r)
    
    print(f"R Shape: {X_r.shape}")
    print(f"BF Shape: {X_BF.shape}")
    print(f"Prediction Parity Error: {pred_error:.4f}")
    assert pred_error < 0.1, f"pred_error={pred_error:.4f} exceeds tolerance 0.1"

def test_model_s_parity():
    print("\n--- Testing Model S Basis Parity ---")
    data_dir = str(_HERE / "BF_data" / "S")
    bird_move = pd.read_csv(_DATA)
    week, lat, species = bird_move['week'].values, bird_move['latitude'].values, bird_move['species'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_BF, S_BF = gam.hgam([week, lat], [5, 5], ["cc", "tp"], species, type="S")
    
    print(f"R Shape: {X_r.shape}")
    print(f"BF Shape: {X_BF.shape}")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_BF, X_r)
    recon_error = np.linalg.norm(X_r - X_BF @ beta_map) / np.linalg.norm(X_r)
    print(f"S Recon Error: {recon_error:.4f}")
    assert recon_error < 0.2, f"recon_error={recon_error:.4f} exceeds tolerance 0.2"

def test_model_i_parity():
    print("\n--- Testing Model I Basis Parity ---")
    data_dir = str(_HERE / "BF_data" / "I")
    bird_move = pd.read_csv(_DATA)
    week, lat, species = bird_move['week'].values, bird_move['latitude'].values, bird_move['species'].values
    X_r = pd.read_csv(f"{data_dir}/X.csv").values
    X_BF, S_BF = gam.hgam([week, lat], [5, 5], ["cc", "tp"], species, type="I")
    
    print(f"R Shape: {X_r.shape}")
    print(f"BF Shape: {X_BF.shape}")
    
    from scipy.linalg import lstsq
    beta_map, res, rank, s = lstsq(X_BF, X_r)
    recon_error = np.linalg.norm(X_r - X_BF @ beta_map) / np.linalg.norm(X_r)
    print(f"I Recon Error: {recon_error:.4f}")
    assert recon_error < 0.2, f"recon_error={recon_error:.4f} exceeds tolerance 0.2"

if __name__ == "__main__":
    test_model_g_parity()
    test_model_gs_parity()
    test_model_gi_parity()
    test_model_s_parity()
    test_model_i_parity()
