import jax.numpy as jnp

# Taxa Mapping (must match load_data.py order)
# 0: Lemur_catta
# 1: Homo_sapiens
# 2: Pan
# 3: Gorilla
# 4: Pongo
# 5: Hylobates
# 6: Macaca_fuscata
# 7: M._mulatta
# 8: M._fascicularis
# 9: M._sylvanus
# 10: Saimiri_sciureus
# 11: Tarsius_syrichta

# Internal Nodes (post-order traversal numbering from 12 to 22)
# Parent idx = N_taxa + internal_idx
# left, right children for each internal node
# 12 is at idx 0 in the arrays

left_children = jnp.array([
    2,  # 12: Pan + Homo(1)
    12, # 13: 12 + Gorilla(3)
    13, # 14: 13 + Pongo(4)
    14, # 15: 14 + Hylobates(5)
    6,  # 16: M_fuscata + M_mulatta(7)
    16, # 17: 16 + M_fascicularis(8)
    17, # 18: 17 + M_sylvanus(9)
    15, # 19: 15 + 18 (Catarrhini)
    19, # 20: 19 + Saimiri(10)
    20, # 21: 20 + Tarsius(11)
    21  # 22: 21 + Lemur(0)
])

right_children = jnp.array([
    1,  # 12
    3,  # 13
    4,  # 14
    5,  # 15
    7,  # 16
    8,  # 17
    9,  # 18
    18, # 19
    10, # 20
    11, # 21
    0   # 22
])

# Branch lengths (approximate genetic distances for testing)
# Total 23 nodes
branch_lengths = jnp.array([
    0.5, 0.05, 0.05, 0.1, 0.2, 0.2, 0.05, 0.05, 0.1, 0.1, 0.3, 0.4, # tips 0-11
    0.02, 0.03, 0.05, 0.05, 0.02, 0.03, 0.05, 0.1, 0.1, 0.1, 0.0  # internal 12-22 (root 22 has 0 bl)
])

def get_tree_data():
    return left_children, right_children, branch_lengths
