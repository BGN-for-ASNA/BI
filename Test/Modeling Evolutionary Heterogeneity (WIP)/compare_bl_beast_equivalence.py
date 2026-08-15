"""
compare_bl_beast_equivalence.py
================================
Comparing BF's latent branch lengths (bl_scale, bl_base) with BEAST-equivalent
metrics (Tree Height and Total Tree Length).

This script extracts node heights and total branch length from the BF posterior
of Model 3 and Model 4 to demonstrate how BF's continuous geometric relaxation
reclaims the geometric uncertainty of BEAST.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
import sys

sys.path.append('..')
from tree_data import get_tree_data

# ── 1. Setup Tree Connectivity ───────────────────────────────────────────────
left_children, right_children, bl_init = get_tree_data()
N_taxa = 12
N_internal = len(left_children)
N_nodes = N_taxa + N_internal

def calculate_tree_metrics(branch_lengths):
    """Calculates Total Length and Tree Height (max distance from root to tip)."""
    # Total length: sum of all 23 branches
    total_length = jnp.sum(branch_lengths)
    
    # Node heights: recursive calculation from tips up
    # However, since these are branch lengths, height_node = height_child + bl
    # For phylograms, we often just use total path from root.
    # We'll calculate the max path from root to any tip.
    # Connectivity is in tree_data: node 22 is root.
    
    node_paths = jnp.zeros(N_nodes)
    # We iterate post-order to build up (or pre-order to push down).
    # Building up distance from root (node 22):
    # root height is 0. tip height is distance from root.
    root_idx = N_nodes - 1
    
    # We'll use a simple BFS/DFS to push distances down.
    distances = {root_idx: 0.0}
    for i in range(N_internal - 1, -1, -1):
        parent = N_taxa + i
        l_c = left_children[i]
        r_c = right_children[i]
        distances[l_c] = distances[parent] + branch_lengths[l_c]
        distances[r_c] = distances[parent] + branch_lengths[r_c]
        
    tip_distances = [distances[i] for i in range(N_taxa)]
    tree_height = max(tip_distances)
    
    return total_length, tree_height

# ── 2. Load BF Samples ────────────────────────────────────────────────────────
# Note: BF_ucln_blmarg_post.csv contains 'bl_scale' as many columns if saved correctly.
# But my fit scripts only saved kappa/alpha/mu/sigma!
# I need to rerun models and save 'bl_scale' samples if I want to compare them.

print("Checking if branch length samples exist in the CSVs...")
try:
    df_m4 = pd.read_csv("Model_4_Temporal_BLMarg/BF_ucln_blmarg_post.csv")
    has_bl = any('bl_scale' in col for col in df_m4.columns)
    if not has_bl:
        print("  WARNING: BF_ucln_blmarg_post.csv does not contain 'bl_scale' columns.")
        print("  I need to update the fit scripts to save the latent branch lengths.")
except:
    print("  ERROR: CSV not found.")

# ── 3. Script to extract from POSTERIOR if I had them ──────────────────────────
# Since I need to rerun, I'll update the fit scripts first.
