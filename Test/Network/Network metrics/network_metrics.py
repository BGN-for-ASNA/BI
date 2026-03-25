from BI import bi
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import networkx as nx
import sys
import os

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Initialize logging to both console and log.txt
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")
log_file = open(log_file_path, "w")
sys.stdout = Tee(sys.stdout, log_file)

m = bi(platform='cpu')

print('Loading data...')
data_path = os.path.dirname(os.path.abspath("")) + "/BI/resources/data/"
G_undirected = nx.karate_club_graph()
G_directed = nx.DiGraph(G_undirected)  # Convert to directed graph

# Get the adjacency matrix for JAX
adj_matrix_np = nx.to_numpy_array(G_directed)
adj_matrix_jax = jnp.array(adj_matrix_np)


def beteeweness_run_all_tests():
    """Iterates through all test configurations and compares JAX vs NetworkX."""
    n_nodes = 15
    seed = 42
    rng = np.random.default_rng(seed)
    
    test_configs = [
        (directed, weighted, normalized)
        for directed in [False, True]
        for weighted in [False, True]
        for normalized in [False, True]
    ]
    
    total_passed = 0
    for directed, weighted, normalized in test_configs:
        print("-" * 70)
        print(f"Testing: Directed={directed}, Weighted={weighted}, Normalized={normalized}")
        
        # 1. Generate graph data
        adj_matrix_np = (rng.random((n_nodes, n_nodes)) < 0.3).astype(np.float32)
        np.fill_diagonal(adj_matrix_np, 0)
        
        weight_matrix_np = None
        if weighted:
            # Generate random weights between 1 and 10
            # Ensure the weight matrix is float32 to match the JAX functions' internal state.
            weight_matrix_np = (rng.uniform(1, 10, size=(n_nodes, n_nodes)) * adj_matrix_np).astype(np.float32)
        
        if not directed:
            adj_matrix_np = np.tril(adj_matrix_np) + np.tril(adj_matrix_np, -1).T
            if weighted:
                weight_matrix_np = np.tril(weight_matrix_np) + np.tril(weight_matrix_np, -1).T

        # 2. Create NetworkX graph
        graph_type = nx.DiGraph if directed else nx.Graph
        G = nx.from_numpy_array(adj_matrix_np, create_using=graph_type)
        
        nx_weight_arg = None
        if weighted:
            nx_weight_arg = 'weight'
            for i, j, data in G.edges(data=True):
                data[nx_weight_arg] = weight_matrix_np[i, j]

        # 3. Run JAX implementation
        adj_matrix_jax = jnp.array(adj_matrix_np)
        weight_matrix_jax = jnp.array(weight_matrix_np) if weighted else None
        
        jax_bc = m.net.betweenness(
            adj_matrix_jax,
            n_nodes=n_nodes,
            weight_matrix=weight_matrix_jax,
            normalized=normalized,
            directed=directed
        ).block_until_ready()

        # 4. Run NetworkX implementation
        nx_bc_dict = nx.betweenness_centrality(
            G,
            normalized=normalized,
            weight=nx_weight_arg
        )
        nx_bc_array = jnp.array([nx_bc_dict[i] for i in range(n_nodes)])

        # 5. Compare results
        try:
            # Use a slightly higher tolerance for float32 vs float64 comparisons
            is_close = jnp.allclose(jax_bc, nx_bc_array, atol=1e-5, rtol=1e-5)
            assert is_close
            print("✅ PASS: JAX and NetworkX results are consistent.")
            total_passed += 1
        except AssertionError:
            print("❌ FAIL: JAX and NetworkX results differ.")
            print(f"   JAX result: {jax_bc}")
            print(f"   NX result:  {nx_bc_array}")

    print("-" * 70)
    print(f"SUMMARY: Passed {total_passed} out of {len(test_configs)} test cases.")


def nx_geodesic(G_undirected, weighted  = True):
    if weighted:
        lengths = dict(nx.all_pairs_dijkstra_path_length(G_undirected))
    else:
        lengths = dict(nx.all_pairs_shortest_path_length(G_undirected))
    nodes = list(G_undirected.nodes)

    # Initialize a distance matrix with 'inf' for unreachable pairs
    n = len(nodes)
    dist_matrix = np.full((n, n), np.inf)

    # Fill the matrix with the shortest path lengths
    for i, source in enumerate(nodes):
        for target, length in lengths[source].items():
            j = nodes.index(target)
            dist_matrix[i, j] = length

    # Set the diagonal to 0 (distance from a node to itself)
    np.fill_diagonal(dist_matrix, 0)

    return dist_matrix


def run_test(name, test_fn):
    """Utility to run a test function and log the result."""
    print(f"Testing {name}------------------------")
    try:
        test_fn()
        print(f"✅ PASS: {name}")
    except Exception as e:
        print(f"❌ FAIL: {name}")
        print(f"   Error: {e}")

run_test("degree (symmetric)", lambda: np.testing.assert_array_almost_equal(
    m.net.degree(adj_matrix_jax,  sym = True), 
    np.array(list(dict(nx.degree(G_undirected)).values()))
))

run_test("degree (asymmetric)", lambda: np.testing.assert_array_almost_equal(
    m.net.degree(adj_matrix_jax,  sym = False), 
    np.array(list(dict(nx.degree(G_undirected)).values()))*2
))

run_test("indegree", lambda: np.testing.assert_array_almost_equal(
    m.net.indegree(adj_matrix_jax,  normalize = True), 
    np.array(list(dict(nx.in_degree_centrality(G_directed)).values()))
))

run_test("outdegree", lambda: np.testing.assert_array_almost_equal(
    m.net.outdegree(adj_matrix_jax,  normalize = True), 
    np.array(list(dict(nx.out_degree_centrality(G_directed)).values()))
))

run_test("strength (symmetric)", lambda: np.testing.assert_array_almost_equal(
    m.net.strength(adj_matrix_jax,  sym = True), 
    np.array(list(dict(G_undirected.degree(weight="weight")).values()))
))

run_test("strength (asymmetric)", lambda: np.testing.assert_array_almost_equal(
    m.net.strength(adj_matrix_jax,  sym = False), 
    np.array(list(dict(G_undirected.degree(weight="weight")).values()))*2
))

run_test("instrength", lambda: np.testing.assert_array_almost_equal(
    m.net.instrength(adj_matrix_jax), 
    np.array(list(dict(G_directed.in_degree(weight="weight")).values()))
))

run_test("outstrength", lambda: np.testing.assert_array_almost_equal(
    m.net.outstrength(adj_matrix_jax), 
    np.array(list(dict(G_directed.out_degree(weight="weight")).values()))
))

run_test("clustering coefficient", lambda: np.testing.assert_array_almost_equal(
    m.net.cc(adj_matrix_jax), 
    np.array(list(dict(nx.clustering(G_directed)).values()))
))

run_test("eigenvector centrality weighted", lambda: np.testing.assert_array_almost_equal(
    m.net.eigenvector(adj_matrix_jax), 
    np.array(list(dict(nx.eigenvector_centrality(G_undirected, max_iter=1000, weight='weight')).values())),
    decimal=5
))

run_test("eigenvector centrality unweighted", lambda: (
    adj_matrix_jax_unweighted := m.net.to_binary_matrix(adj_matrix_jax),
    np.testing.assert_array_almost_equal(
        m.net.eigenvector(adj_matrix_jax_unweighted), 
        np.array(list(dict(nx.eigenvector_centrality(G_undirected, max_iter=1000)).values())),
        decimal=5
    )
)[1])

run_test("density", lambda: np.testing.assert_array_almost_equal(
    m.net.density(adj_matrix_jax), 
    np.array(nx.density(G_undirected))
))

run_test("diameter", lambda: np.testing.assert_array_almost_equal(
    m.net.diameter(m.net.to_binary_matrix(adj_matrix_jax)), 
    np.array(nx.diameter(G_undirected))
))

run_test("geodesic_distance weighted", lambda: np.testing.assert_array_almost_equal(
    m.net.geodesic_distance(m.net.to_binary_matrix(adj_matrix_jax)), 
    np.array(nx_geodesic(G_undirected, weighted=False))
))

run_test("geodesic_distance binary", lambda: np.testing.assert_array_almost_equal(
    m.net.geodesic_distance(adj_matrix_jax), 
    np.array(nx_geodesic(G_undirected, weighted=True))
))

beteeweness_run_all_tests()

print("\nAll tests completed.")
log_file.close()