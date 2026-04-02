import os
from Bio import Phylo
import numpy as np

def parse_beast_trees(tree_file, num_samples=10, burnin=50):
    trees = list(Phylo.parse(tree_file, "nexus"))
    # Apply burnin and uniform sampling
    trees = trees[burnin:]
    step = max(len(trees) // num_samples, 1)
    sampled_trees = trees[::step][:num_samples]
    
    # Assuming all trees have the same taxa names
    taxa_names = [clade.name for clade in sampled_trees[0].get_terminals()]
    taxa_to_idx = {name: i for i, name in enumerate(taxa_names)}
    
    N_taxa = len(taxa_names)
    num_internal = N_taxa - 1
    N_nodes = N_taxa + num_internal
    
    left_children = np.zeros((num_samples, num_internal), dtype=int)
    right_children = np.zeros((num_samples, num_internal), dtype=int)
    branch_lengths = np.zeros((num_samples, N_nodes), dtype=float)
    
    # We assign indices such that 0..N-1 are tips, N..2N-2 are internal nodes (topological sorting post-order)
    for k, tree in enumerate(sampled_trees):
        # assign indices post-order
        current_internal = N_taxa
        idx_map = {}
        for clade in tree.find_clades(order="postorder"):
            if clade.is_terminal():
                idx = taxa_to_idx[clade.name]
                idx_map[clade] = idx
            else:
                idx = current_internal
                current_internal += 1
                idx_map[clade] = idx
                
                # Children must be precisely 2 for strictly bifurcating trees
                children = clade.clades
                left_children[k, idx - N_taxa] = idx_map[children[0]]
                right_children[k, idx - N_taxa] = idx_map[children[1]]
                
            # Branch lengths
            # Root branch length is usually None or 0.0
            if clade.branch_length is not None:
                branch_lengths[k, idx] = clade.branch_length
            else:
                branch_lengths[k, idx] = 0.0

    return taxa_names, left_children, right_children, branch_lengths

if __name__ == "__main__":
    t_file = "../test.time.123.trees"
    names, L, R, BL = parse_beast_trees(t_file, num_samples=5, burnin=200)
    print("Parsed", len(names), "taxa")
    print("Left mapping shape:", L.shape)
    print("Branch length shape:", BL.shape)
