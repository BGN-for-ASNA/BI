import jax
import jax.numpy as jnp
from jax import jit, lax
from jax import vmap
from functools import partial
from typing import Optional, Tuple

class met:
    """Network metrics class for computing various graph metrics using JAX.
    This class provides methods to compute clustering coefficients, eigenvector centrality, Dijkstra's algorithm for shortest paths, and other network metrics. 
    It leverages JAX's capabilities for efficient computation on large graphs.
    """
    def __init__(self):
        init_betweenness = jnp.zeros((4,4))  
        met.betweenness(init_betweenness, 4)
    
    @jit
    def normalize(x, m):
        return x / (m.shape[0]-1)
        
    # Network utils
    # Nodal measures----------------------------------------------------------------------------------
    ## Clustering_coefficient----------------------------------------------------------------------------------
    @staticmethod 
    @jax.jit
    def triangles_and_degree(adj_matrix):
        """
        Computes the number of triangles and the degree for each node in    a graph.
        This function is optimized for JAX and is JIT-compatible.
        """
        # Ensure the adjacency matrix is boolean for logical operations
        adj_matrix_bool = adj_matrix > 0

        # Calculate the degree of each node
        degrees = jnp.sum(adj_matrix_bool, axis=1)

        # Compute the number of triangles for each node
        # This is equivalent to the diagonal of the cube of the     adjacency matrix
        num_triangles = jnp.diag(jnp.linalg.matrix_power(adj_matrix_bool.   astype(jnp.int32), 3)) / 2

        return degrees, num_triangles

    @jax.jit
    def clustering_coefficient(adj_matrix):
        """
        Computes the clustering coefficient for each node in the graph.
        This function is optimized for JAX and is JIT-compatible.
        """
        degrees, num_triangles = met.triangles_and_degree(adj_matrix)

        # To avoid division by zero for nodes with degree less than 2,
        # we calculate the denominator and use jnp.where to handle these    cases.
        denominator = degrees * (degrees - 1)

        # The clustering coefficient is set to 0 where the denominator  is 0.
        clusterc = jnp.where(denominator > 0, 2 * num_triangles /   denominator, 0)

        return clusterc

    @staticmethod 
    def cc(m, nodes=None):
        return met.clustering_coefficient(m) 

    ## eigenvector----------------------------------------------------------------------------------
    @staticmethod
    @jit
    def power_iteration(A):
        A = jnp.asarray(A, dtype=jnp.float64)
        n = A.shape[0]
        if n == 0:
            return jnp.array([], dtype=jnp.float64)
        v0 = jnp.ones(n, dtype=jnp.float64) / jnp.sqrt(n)
        v = lax.fori_loop(0, 100, lambda i, v_prev: jnp.where(jnp.linalg.norm(A @ v_prev) > 0, (A @ v_prev) / jnp.linalg.norm(A @ v_prev), A @ v_prev), v0)
        max_abs_idx = jnp.argmax(jnp.abs(v))
        sign = jnp.sign(v[max_abs_idx])
        v = v * jnp.where(sign == 0, 1.0, sign)
        return v

    @staticmethod
    @jit
    def _jax_bfs_component_mask(A, start_node):
        """
        Finds the connected component containing start_node using a JAX-native BFS.
        """
        n = A.shape[0]
        # `visited` tracks all nodes in the component. `frontier` is the current layer.
        visited = jnp.zeros(n, dtype=bool).at[start_node].set(True)
        frontier = visited.copy()

        def cond_fn(state):
            # Continue as long as there are nodes in the frontier to expand
            visited, frontier = state
            return frontier.any()

        def body_fn(state):
            visited, frontier = state
            # Find all direct neighbors of the current frontier
            new_frontier = ((A @ frontier.astype(A.dtype)) > 0) & (~visited)
            # Add the new neighbors to the visited set for the next iteration
            new_visited = visited | new_frontier
            return new_visited, new_frontier

        # The loop runs until no new nodes can be reached
        final_visited, _ = lax.while_loop(cond_fn, body_fn, (visited, frontier))
        return final_visited

    @staticmethod
    @partial(jit, static_argnames=['use_transpose'])
    def eigenvector(A, use_transpose=True):
        """
        Fully JIT-compiled eigenvector centrality using an efficient JAX-native BFS.
        """
        def for_undirected(A):
            # For undirected graphs, any non-zero node is a good starting point.
            # We use degree as a heuristic to likely start in a large component.
            start_node = jnp.argmax(A.sum(axis=1))
            
            # 1. Get mask for the largest component using the efficient JAX BFS
            mask = met._jax_bfs_component_mask(A, start_node)
            
            # 2. Apply mask to isolate the sub-matrix (no change from before)
            sub_matrix = A * mask[:, None] * mask[None, :]
            
            # 3. Run power iteration
            centrality = met.power_iteration(sub_matrix)
            return centrality

        def for_directed(A):
            M = A.T if use_transpose else A
            return met.power_iteration(M)

        is_undirected = jnp.all(A == A.T)
        return lax.cond(is_undirected, for_undirected, for_directed, A)

    ## Dijkstra----------------------------------------------------------------------------------
    @staticmethod 
    @jit
    def dijkstra_jax(adjacency_matrix, source):
        """
        Compute the shortest path from a source node to all other nodes using Dijkstra's algorithm.

        Dijkstra's algorithm finds the shortest paths between nodes in a graph, particularly useful
        for graphs with non-negative edge weights. This function uses JAX for efficient computation.

        Parameters:
        -----------
        adjacency_matrix : jax.numpy.ndarray
            A square (n x n) adjacency matrix representing the graph. The element at (i, j)
            represents the weight of the edge from node i to node j. Non-zero values indicate
            a connection, and higher values indicate longer paths.

        source : int
            The index of the source node from which the shortest paths are computed.

        Returns:
        --------
        jax.numpy.ndarray
            A 1D array of length n where each element represents the shortest distance from the
            source node to the corresponding node. The source node will have a distance of 0.

        """
        n = adjacency_matrix.shape[0]
        visited = jnp.zeros(n, dtype=bool)
        dist = jnp.inf * jnp.ones(n)
        dist = dist.at[source].set(0)

        def body_fn(carry):
            visited, dist = carry

            # Find the next node to process
            u = jnp.argmin(jnp.where(visited, jnp.inf, dist))
            visited = visited.at[u].set(True)

            # Update distances to all neighbors
            def update_dist(v, dist):
                return jax.lax.cond(
                    jnp.logical_and(jnp.logical_not(visited[v]), adjacency_matrix[u, v] > 0),
                    lambda _: jnp.minimum(dist[v], dist[u] + adjacency_matrix[u, v]),
                    lambda _: dist[v],
                    None
                )

            dist = lax.fori_loop(0, n, lambda v, dist: dist.at[v].set(update_dist(v, dist)), dist)

            return visited, dist

        def cond_fn(carry):
            visited, _ = carry
            return jnp.any(jnp.logical_not(visited))

        # Loop until all nodes are visited
        visited, dist_final = lax.while_loop(cond_fn, body_fn, (visited, dist))

        return dist_final

    @staticmethod 
    def dijkstra(m,  source):
        return met.dijkstra_jax(m, source)
    

    ## Strength----------------------------------------------------------------------------------
    @staticmethod 
    @jit    
    def outstrength_jit(x):
        return jnp.sum(x, axis=1)

    @staticmethod 
    @jit
    def instrength_jit(x):
        return jnp.sum(x, axis=0)

    @staticmethod 
    @jit
    def strength_jit(x):
        return met.outstrength_jit(x) +  met.instrength_jit(x)

    @staticmethod 
    def strength(m, sym = False):
        if sym :
            return met.outstrength_jit(m)
        else:
            return met.strength_jit(m)

    
    @staticmethod 
    def outstrength(m):
        return met.outstrength_jit(m)
    
    @staticmethod 
    def instrength(m):
        return met.instrength_jit(m)

    ## Degree----------------------------------------------------------------------------------
    @staticmethod 
    @jit
    def outdegree_jit(x):
        mask = x != 0
        return jnp.sum(mask, axis=1)

    @staticmethod 
    @jit
    def indegree_jit(x):
        mask = x != 0
        return jnp.sum(mask, axis=0)

    @staticmethod 
    @jit
    def degree_jit(x):
        return met.indegree_jit(x) + met.outdegree_jit(x)

    @staticmethod 
    def degree(m, sym = False, normalize=False):
        # normalized by dividing by the maximum possible degree in a simple graph n-1 where n is the number of nodes in G.
        if sym :
            degree =  met.indegree_jit(m)
        else:
            degree = met.degree_jit(m)

        if normalize:
            return met.normalize(degree,m)
        else:
            return degree
    
    @staticmethod 
    def indegree(m, normalize=False):
        degree =  met.indegree_jit(m)
        if normalize:
            return met.normalize(degree,m)
        else:
            return degree
    
    @staticmethod 
    def outdegree(m, normalize=False):
        degree = met.outdegree_jit(m)
        if normalize:
            return met.normalize(degree,m)
        else:
            return degree
    
    # Global measures----------------------------------------------------------------------------------
    @staticmethod
    @jit
    def density(m):
        """
        Compute the network density from the weighted adjacency matrix.

        Args:
            adj_matrix: JAX array representing the weighted adjacency matrix of a graph.

        Returns:
            Network density as a float.
        """
        n_nodes = m.shape[0]
        n_possible_edges = n_nodes * (n_nodes - 1) / 2
        n_actual_edges = jnp.count_nonzero(m) / 2  # Since the matrix is symmetric

        # Density formula
        density = n_actual_edges / n_possible_edges
        return density

    @staticmethod
    @jit
    def single_source_dijkstra(m, src):
        """
        Computes the shortest path from a source node to all other nodes
        in a weighted graph using Dijkstra's algorithm.
        """
        n_nodes = m.shape[0]
        
        # Initialize distances and visited status
        dist = jnp.full((n_nodes,), jnp.inf).at[src].set(0)
        visited = jnp.zeros((n_nodes,), dtype=bool)

        def relax_step(carry, _):
            dist, visited = carry
            
            # Find the closest unvisited node
            unvisited_dist = jnp.where(visited, jnp.inf, dist)
            u = jnp.argmin(unvisited_dist)
            visited = visited.at[u].set(True)
            
            # Relax distances for neighbors of the selected node
            new_dist = jnp.minimum(dist, dist[u] + m[u])
            return (new_dist, visited), None

        # The loop runs n_nodes times to ensure all nodes are visited
        (dist, _), _ = jax.lax.scan(relax_step, (dist, visited), None, length=n_nodes)

        return dist

    @staticmethod
    @jit
    def geodesic_distance(m):
        """
        Compute the geodesic distance in a weighted graph using Dijkstra's algorithm in JAX.
        Args:
            m: 2D JAX array representing the weighted adjacency matrix of a graph.

        Returns:
            A 2D JAX array containing the shortest path distances between all pairs of nodes.
        """
        # Replace 0s with infinity for non-existent edges, but keep diagonal as 0
        m = jnp.where(m == 0, jnp.inf, m)
        m = m.at[jnp.diag_indices_from(m)].set(0)
        
        n_nodes = m.shape[0]

        # Use vmap to run Dijkstra from each node as a source.
        # in_axes=(None, 0) means the first argument (m) is broadcasted (the same for all calls)
        # and the second argument (the source nodes) is mapped over.
        distances = jax.vmap(met.single_source_dijkstra, in_axes=(None, 0))(m, jnp.arange(n_nodes))
        return distances
    
    @staticmethod
    @jit
    def diameter(m):
        """
        Compute the diameter of a graph using the geodesic distance.
        Args:
            adj_matrix: 2D JAX array representing the weighted adjacency matrix of a graph. 
            
        Returns:
            The diameter of the graph.
        """
        return jnp.max(met.geodesic_distance(m))


    # Betweenness centrality  ----------------------------------------------------------------------------------

    # --- Paste your helper functions here ---
    @staticmethod
    @partial(jit, static_argnames=['n_nodes'])
    def dijkstra(
        adjacency_matrix: jnp.ndarray,
        weight_matrix: jnp.ndarray,
        source: int,
        n_nodes: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ Breadth-First Search (BFS)"""
        initial_dist = jnp.full(n_nodes, jnp.inf, dtype=jnp.float32).at[source].set(0.0)
        initial_sigma = jnp.zeros(n_nodes, dtype=jnp.float32).at[source].set(1.0)
        initial_P = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float32)
        initial_visited = jnp.zeros(n_nodes, dtype=bool)
        initial_S = jnp.full(n_nodes, -1, dtype=jnp.int32)
        initial_s_idx = 0
        initial_state = (initial_dist, initial_sigma, initial_P, initial_visited, initial_S, initial_s_idx)
        def body_fun(_, state):
            dist, sigma, P_matrix, visited, S, s_idx = state
            unvisited_dist = jnp.where(visited, jnp.inf, dist)
            u = jnp.argmin(unvisited_dist)
            min_dist_u = unvisited_dist[u]
            do_update = min_dist_u < jnp.inf
            S = S.at[s_idx].set(jnp.where(do_update, u, S[s_idx]))
            s_idx = jnp.where(do_update, s_idx + 1, s_idx)
            visited = visited.at[u].set(jnp.where(do_update, True, visited[u]))
            new_dist_v = dist[u] + weight_matrix[u, :]
            is_neighbor = adjacency_matrix[u, :] > 0
            is_unvisited = ~visited
            shorter_path_found = do_update & is_neighbor & is_unvisited & (new_dist_v < dist)
            equal_path_found = do_update & is_neighbor & is_unvisited & (jnp.abs(new_dist_v - dist) < 1e-9)
            dist = jnp.where(shorter_path_found, new_dist_v, dist)
            sigma_after_reset = jnp.where(shorter_path_found, 0.0, sigma)
            sigma_to_add = jnp.where(shorter_path_found | equal_path_found, sigma[u], 0.0)
            sigma = sigma_after_reset + sigma_to_add
            P_matrix = jnp.where(shorter_path_found[:, jnp.newaxis], 0.0, P_matrix)
            predecessor_update_mask = shorter_path_found | equal_path_found
            new_col_u = jnp.where(predecessor_update_mask, 1.0, P_matrix[:, u])
            P_matrix = P_matrix.at[:, u].set(new_col_u)
            return dist, sigma, P_matrix, visited, S, s_idx
        _, final_sigma, final_P, _, final_S, final_s_idx = lax.fori_loop(0, n_nodes, body_fun, initial_state)
        output_mask = jnp.arange(n_nodes) < final_s_idx
        gather_indices = jnp.maximum(0, final_s_idx - 1 - jnp.arange(n_nodes))
        reversed_S_padded = final_S[gather_indices]
        S_reversed = jnp.where(output_mask, reversed_S_padded, -1)
        return S_reversed, final_P, final_sigma

    @staticmethod
    @partial(jit, static_argnames=['n_nodes'])
    def bfs(adjacency_matrix, source, n_nodes):
        """ Breadth-First Search (BFS)"""
        dist = jnp.full(n_nodes, -1, dtype=jnp.int32).at[source].set(0)
        sigma = jnp.zeros(n_nodes, dtype=jnp.float32).at[source].set(1.0)
        P_matrix = jnp.zeros((n_nodes, n_nodes), dtype=jnp.float32)
        layer_mask = (dist == 0)
        def body_fun(i, state):
            dist, sigma, P_matrix, layer_mask = state
            neighbors_matrix = jnp.where(layer_mask[:, None], adjacency_matrix, 0)
            potential_next_layer = (neighbors_matrix.sum(axis=0) > 0)
            newly_discovered_mask = potential_next_layer & (dist == -1)
            dist = jnp.where(newly_discovered_mask, i + 1, dist)
            is_in_next_layer_mask = (dist == i + 1)
            predecessor_mask = (adjacency_matrix.T > 0) & layer_mask[None, :]
            predecessor_sigmas = jnp.where(predecessor_mask, sigma[None, :], 0)
            sigma_contribution = predecessor_sigmas.sum(axis=1)
            sigma = jnp.where(is_in_next_layer_mask, sigma + sigma_contribution, sigma)
            P_update = jnp.where(is_in_next_layer_mask[None, :] & layer_mask[:, None] & (adjacency_matrix > 0), 1.0,    0)
            P_matrix = P_matrix + P_update.T
            return dist, sigma, P_matrix, is_in_next_layer_mask
        dist, sigma, P_matrix, _ = lax.fori_loop(0, n_nodes, body_fun, (dist, sigma, P_matrix, layer_mask))
        S = jnp.flip(jnp.argsort(dist, stable=True))
        return S, P_matrix, sigma, dist

    @staticmethod
    @partial(jit, static_argnames=['n_nodes'])
    def _optimized_accumulate_basic(betweenness, S, P, sigma, source, n_nodes):
        initial_delta = jnp.zeros(n_nodes, dtype=jnp.float32)
        initial_state = (betweenness, initial_delta)
        def body_fun(i, state):
            betweenness, delta = state
            w = S[i]
            predecessors_mask = P[w, :] > 0
            coeff = (sigma / sigma[w]) * (1.0 + delta[w])
            delta_update = jnp.where(predecessors_mask, coeff, 0.0)
            delta_new = delta + delta_update
            betweenness_update = jnp.where(w == source, 0.0, delta[w])
            betweenness_new = betweenness.at[w].add(betweenness_update)
            return (betweenness_new, delta_new)
        final_betweenness, _ = lax.fori_loop(0, n_nodes, body_fun, initial_state)
        return final_betweenness

    @staticmethod
    @partial(jit, static_argnames=['n_nodes', 's_len'])
    def _optimized_accumulate_endpoints(betweenness, S, P, sigma, source, n_nodes, s_len):
        betweenness = betweenness.at[source].add(s_len - 1)
        initial_delta = jnp.zeros(n_nodes, dtype=jnp.float32)
        initial_state = (betweenness, initial_delta)
        def body_fun(i, state):
            betweenness, delta = state
            w = S[i]
            betweenness_update_val = delta[w] + 1
            betweenness = betweenness.at[w].add(jnp.where(w != source, betweenness_update_val, 0.0))
            predecessors_mask = P[w, :] > 0
            safe_sigma_w = jnp.where(sigma[w] == 0, 1.0, sigma[w])
            coeff = (sigma / safe_sigma_w) * (1.0 + delta[w])
            is_valid_update = predecessors_mask & (sigma[w] > 0)
            delta_update = jnp.where(is_valid_update, coeff, 0.0)
            delta = delta + delta_update
            return (betweenness, delta)
        final_betweenness, _ = lax.fori_loop(0, s_len, body_fun, initial_state)
        return final_betweenness

    @staticmethod
    @jit
    def _rescale_jax(betweenness, n_nodes, normalized, k, endpoints, n_sampled, directed=False):
        def scale_fn(_):
            def small_graph(_):
                return betweenness
            def normal_graph(_):
                def endpoints_scale(_):
                    return 1.0 / ((n_nodes - 1) * (n_nodes - 2))
                def directed_or_undirected(_):
                    return lax.cond(directed, lambda _: 1.0 / ((n_nodes - 1) * (n_nodes - 2)), lambda _: 2.0 /  ((n_nodes - 1) * (n_nodes - 2)), operand=None)
                scale = lax.cond(endpoints, endpoints_scale, directed_or_undirected, operand=None)
                scale = lax.cond((k is not None) & (n_sampled < n_nodes), lambda s: s * n_nodes / n_sampled, lambda     s: s, operand=scale)
                return betweenness * scale
            return lax.cond(n_nodes <= 2, small_graph, normal_graph, operand=None)
        return lax.cond(normalized, scale_fn, lambda _: betweenness, operand=None)


    @staticmethod
    @partial(jit, static_argnames=['n_nodes', 'endpoints', 'use_weights'])
    def _vmapped_betweenness_computation(
        adjacency_matrix: jnp.ndarray,
        weight_matrix: Optional[jnp.ndarray],
        source_nodes: jnp.ndarray,
        n_nodes: int,
        endpoints: bool,
        use_weights: bool
    ) -> jnp.ndarray:
        def single_source_logic(s):
            b_s = jnp.zeros(n_nodes, dtype=jnp.float32)
            if use_weights:
                S, P, sigma = met.dijkstra(adjacency_matrix, weight_matrix, s, n_nodes=n_nodes)
            else:
                S, P, sigma, _ = met.bfs(adjacency_matrix, s, n_nodes=n_nodes)
            s_len = (S != -1).sum()
            if endpoints:
                b_s = met._optimized_accumulate_endpoints(b_s, S, P, sigma, s, n_nodes, s_len)
            else:
                b_s = met._optimized_accumulate_basic(b_s, S, P, sigma, s, n_nodes)
            return b_s
        all_b_s = vmap(single_source_logic)(source_nodes)
        return all_b_s.sum(axis=0)

    # --- CORRECTED MAIN FUNCTION ---
    @staticmethod
    def betweenness(
        adjacency_matrix: jnp.ndarray,
        n_nodes: int,
        k: int = None,
        weight_matrix: jnp.ndarray = None,
        normalized: bool = True,
        endpoints: bool = False,
        directed: bool = False,
    ) -> jnp.ndarray:
        if k is not None:
            raise NotImplementedError("Sampling (k < n_nodes) is not implemented in this example.")

        source_nodes = jnp.arange(n_nodes)
        n_sampled = n_nodes

        betweenness = met._vmapped_betweenness_computation(
            adjacency_matrix,
            weight_matrix,
            source_nodes,
            n_nodes=n_nodes,
            endpoints=endpoints,
            use_weights=(weight_matrix is not None)
        )

        # Correct for double-counting in undirected graphs.

        if not directed:
            betweenness /= 2.0

        betweenness = met._rescale_jax(
            betweenness, n_nodes, normalized, k, endpoints, n_sampled, directed
        )
        return betweenness
    