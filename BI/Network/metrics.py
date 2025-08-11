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
    

    # Network utils
    # Nodal measures----------------------------------------------------------------------------------
    ## Clustering_coefficient----------------------------------------------------------------------------------
    @staticmethod 
    def triangles_and_degree_iter(adj_matrix, nodes=None):
        """Compute triangles and degrees for nodes in the graph."""
        num_nodes = adj_matrix.shape[0]
        if nodes is None:
            nodes = jnp.arange(num_nodes)
        results = []

        for v in nodes:
            v_nbrs = jnp.where(adj_matrix[v] > 0)[0]
            vs = jnp.setdiff1d(v_nbrs, jnp.array([v]))
            gen_degree = jnp.array([
                jnp.sum(jnp.logical_and(adj_matrix[w], adj_matrix[v])) for w in vs
            ])
            ntriangles = jnp.sum(gen_degree)
            results.append((v, len(vs), ntriangles, gen_degree))
        return results

    @staticmethod 
    def clustering(adj_matrix, nodes=None):
        """Compute unweighted clustering coefficient for nodes in the graph."""
        td_iter = met.triangles_and_degree_iter(adj_matrix, nodes)
        num_nodes = adj_matrix.shape[0]
        clusterc = jnp.zeros(num_nodes)

        for v, d, t, _ in td_iter:
            clusterc = clusterc.at[v].set(0 if t == 0 else t / (d * (d - 1)))
        return clusterc


    @staticmethod 
    def cc(m, nodes=None):
        return met.clustering(m, nodes=nodes) 

    ## eigenvector----------------------------------------------------------------------------------
    @staticmethod 
    @jit
    def power_iteration(m, num_iter=100, tol=1e-6):
        """
        Compute the dominant eigenvector of a matrix A  using the power iteration algorithm.

        Args:
            m (jax.numpy.ndarray): Input square matrix.
            num_iter (int): Maximum number of   iterations.
            tol (float): Tolerance for convergence.

        Returns:
            eigenvector (jax.numpy.ndarray): Dominant   eigenvector.
            eigenvalue (float): Dominant eigenvalue.
        """
        def cond_fn(state):
            # Condition to continue iterating
            i, b_k, b_k1, _ = state
            return (i < num_iter) & (jnp.linalg.norm(b_k1 - b_k) >= tol)

        def body_fn(state):
            # Perform one step of the power iteration
            i, b_k, b_k1, m = state
            b_k = b_k1
            b_k1 = jnp.dot(m, b_k)
            b_k1 = b_k1 / jnp.linalg.norm(b_k1)
            return (i + 1, b_k, b_k1, m)

        # Initialize the state
        n = m.shape[0]
        b_k = jnp.ones(n)
        b_k1 = jnp.dot(m, b_k)
        b_k1 = b_k1 / jnp.linalg.norm(b_k1)
        state = (0, b_k, b_k1, m)

        # Run the while loop
        final_state = lax.while_loop(cond_fn, body_fn,  state)
        _, _, b_k1, _ = final_state

        # Compute the dominant eigenvalue
        eigenvalue = jnp.dot(b_k1, jnp.dot(m, b_k1)) /  jnp.dot(b_k1, b_k1)
        return b_k1, eigenvalue

    @staticmethod
    @jit
    def eigenvector(m, num_iter=300, tol=1e-6):
        """
        Compute the eigenvector centrality of a graph using the power iteration algorithm.

        Args:
            m (jax.numpy.ndarray): Input square matrix.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for convergence.

        Returns:
            eigenvector_centrality (jax.numpy.ndarray): Eigenvector centrality of each node.
        """
        eigenvector, eigenvalue = met.power_iteration(m, num_iter=num_iter, tol=tol)
        return eigenvector

    ## Dijkstra----------------------------------------------------------------------------------
    @staticmethod 
    @jit
    def dijkstra(adjacency_matrix, source):
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
        return met.dijkstra(m, source)
    

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
    def strength(m):
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
        return met.indegree_jit(x) +met.outdegree_jit(x)

    @staticmethod 
    def degree(m):
        return met.degree_jit(m)
    
    @staticmethod 
    def indegree(m):
        return met.indegree_jit(m)
    
    @staticmethod 
    def outdegree(m):
        return met.outdegree_jit(m)
    
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
