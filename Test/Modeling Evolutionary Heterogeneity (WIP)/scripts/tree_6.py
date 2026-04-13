import jax.numpy as jnp

# Taxa (6 taxa subset)
# 0: human
# 1: chimp
# 2: bonobo
# 3: gorilla
# 4: orangutan
# 5: siamang

# Tree structure (UPGMA fits to 768 sites)
# Post-order internal nodes: 6, 7, 8, 9, 10 (Root)
# Internal node indices in arrays: node_idx - 6

left_children = jnp.array([
    1,  # 6: chimp + bonobo
    0,  # 7: human + Node 6
    3,  # 8: gorilla + Node 7
    4,  # 9: orangutan + Node 8
    5   # 10: siamang + Node 9
])

right_children = jnp.array([
    2,  # 6
    6,  # 7
    7,  # 8
    8,  # 9
    9   # 10
])

# Branch lengths (from UPGMA)
# Tips 0-5, Internal 6-10
branch_lengths = jnp.array([
    0.01855, # 0: human (to 7)
    0.00845, # 1: chimp (to 6)
    0.00845, # 2: bonobo (to 6)
    0.02600, # 3: gorilla (to 8)
    0.04425, # 4: orangutan (to 9)
    0.05180, # 5: siamang (to 10)
    0.01010, # 6: (to 7)
    0.00745, # 7: (to 8)
    0.01825, # 8: (to 9)
    0.00755, # 9: (to 10)
    0.00000  # 10: root
])

def get_tree_data():
    return left_children, right_children, branch_lengths
