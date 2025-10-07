import jax
import jax.numpy as jnp
from jax import jit, lax
from jax import vmap

#region Class <comment>
class met:
    """Network metrics class for computing various graph metrics using JAX.
    This class provides methods to compute clustering coefficients, eigenvector centrality, Dijkstra's algorithm for shortest paths, and other network metrics. 
    It leverages JAX's capabilities for efficient computation on large graphs.
    """
    def __init__(self):
        pass
    
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
    def power_iteration(A, num_iter=1000, tol=1e-6):
        # ensure float64 to match numpy/networkx precision
        A = jnp.asarray(A, dtype=jnp.float64)
        n = A.shape[0]

        # start with normalized vector of ones (float64)
        v0 = jnp.ones(n, dtype=jnp.float64)
        v0 = v0 / jnp.linalg.norm(v0)

        def cond_fn(state):
            i, prev_v = state
            v = jnp.dot(A, prev_v)
            v = v / jnp.linalg.norm(v)
            return (i < num_iter) & (jnp.linalg.norm(v - prev_v) >= tol)

        def body_fn(state):
            i, prev_v = state
            v = jnp.dot(A, prev_v)
            v = v / jnp.linalg.norm(v)
            return (i + 1, v)

        init_state = (0, v0)
        _, v = lax.while_loop(cond_fn, body_fn, init_state)

        # Rayleigh quotient for eigenvalue (real-valued)
        eigenvalue = jnp.dot(v, jnp.dot(A, v)) / jnp.dot(v, v)
        return v, eigenvalue

    @staticmethod
    def eigenvector(adj_matrix, weighted=True, use_transpose=True,
                    add_self_loops=False, num_iter=1000, tol=1e-6):
        """
        Compute eigenvector centrality. Arguments to match NetworkX behavior:
         - use_transpose: set True for iterating with A.T (incoming edges convention).
         - add_self_loops: set True only if you intentionally want self-loops.
        """
        A = jnp.asarray(adj_matrix, dtype=jnp.float64)

        if not weighted:
            A = (A > 0).astype(jnp.float64)

        M = A.T if use_transpose else A
        if add_self_loops:
            M = M + jnp.eye(M.shape[0], dtype=jnp.float64)

        v, _ = met.power_iteration(M, num_iter=num_iter, tol=tol)
        # normalize to L2 (NetworkX normalizes similarly)
        v = v / jnp.linalg.norm(v)
        return v
    
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
    def single_source_dijkstra(src):
        # Initialize distances and visited status
        dist = jnp.full((n_nodes,), jnp.inf)
        dist = dist.at[src].set(0)
        visited = jnp.zeros((n_nodes,), dtype=bool)

        def relax_step(carry, _):
            dist, visited = carry
            # Find the closest unvisited node
            unvisited_dist = jnp.where(visited, jnp.inf, dist)
            u = jnp.argmin(unvisited_dist)
            visited = visited.at[u].set(True)
            # Relax distances for neighbors of the selected node
            new_dist = jnp.where(
                ~visited,
                jnp.minimum(dist, dist[u] + m[u]),
                dist
            )
            return (new_dist, visited), None

        (dist, _), _ = jax.lax.scan(relax_step, (dist, visited), None, length=n_nodes)

        return dist

    @staticmethod
    def geodesic_distance(m):
        """
        Compute the geodesic distance in a weighted graph using Dijkstra's algorithm in JAX.
        Args:
            adj_matrix: 2D JAX array representing the weighted adjacency matrix of a graph.

        Returns:
            A 2D JAX array containing the shortest path distances between all pairs of  nodes.
        """
        m=m.at[jnp.where(m == 0)].set(jnp.inf)
        n_nodes = m.shape[0]



        distances = jax.vmap(met.single_source_dijkstra)(jnp.arange(n_nodes))
        return distances

    @staticmethod
    def diameter(m):
        """
        Compute the diameter of a graph using the geodesic distance.
        Args:
            adj_matrix: 2D JAX array representing the weighted adjacency matrix of a graph. 
            
        Returns:
            The diameter of the graph.
        """
        return jnp.max(met.geodesic_distance(m))
