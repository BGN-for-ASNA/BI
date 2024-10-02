import jax
import jax.numpy as jnp
from jax import jit, lax

#region Network_metrics

#tag clustering_coefficient 
def clustering_coefficient(adjacency_matrix):
    """ Compute the clustering coefficient for each nodes of a graph.
        For each node v, the clustering coefficient is calculated as:

        𝐶(𝑣)=2× number of triangles connected to v / degree(𝑣) × (degree(𝑣)−1)
        Where:

        A triangle is formed when a node's neighbors are also connected to each other.
        The degree of a node is the number of direct neighbors.

    Args:
        adjacency_matrix (Squared 2D jax array): Adjacency matrix of the graph.

    Returns:
        _type_: jax 1D array of clustering coefficient for each node.
    """

    degrees = jnp.sum(adjacency_matrix, axis=1)
    triangles = jnp.einsum('ij,jk,ki->i', adjacency_matrix, adjacency_matrix, adjacency_matrix) / 2
    possible_triangles = degrees * (degrees - 1)
    clustering = jnp.where(possible_triangles > 0, triangles / possible_triangles, 0)
    return clustering * 2

#tag Eigenvector <comment>
@jit
def eigenvector_centrality(adjacency_matrix, tol=1e-6, max_iter=100):
    """
    Compute the eigenvector centrality of a network using power iteration.

    Eigenvector centrality is a measure of the influence of a node in a network. Nodes
    that are connected to many highly connected nodes will have higher centrality.

    Parameters:
    -----------
    adjacency_matrix : jax.numpy.ndarray
        A square (n x n) adjacency matrix representing the graph. The element at (i, j)
        represents the weight of the edge from node i to node j.
        
    tol : float, optional, default=1e-6
        The tolerance for convergence. The iteration will stop when the difference between
        successive iterations is smaller than this threshold.
        
    max_iter : int, optional, default=100
        The maximum number of iterations allowed to achieve convergence.

    Returns:
    --------
    jax.numpy.ndarray
        A 1D array of length n representing the eigenvector centrality of each node.
        The values are normalized to unit length.
    """

    x = jnp.ones(adjacency_matrix.shape[0])
    
    def body_fn(carry):
        x, _ = carry
        # Matrix-vector multiplication
        x_new = jnp.dot(adjacency_matrix, x)
        # Normalize
        x_new = x_new / jnp.linalg.norm(x_new)
        return x_new, x
        
    # Check for convergence
    def cond_fn(carry):
        x_new, x = carry
        diff = jnp.linalg.norm(x_new - x)
        return diff >= tol

    # Initialize the loop with x and a dummy previous x (e.g., zeros)
    x_final, _ = lax.while_loop(cond_fn, body_fn, (x, x + tol + 1))

    return x_final

#tag dijkstra <comment>
jit
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

#tag outstrength
@jit    
def outstrength_jit(x):
    return jnp.sum(x, axis=1)

#tag instrength
@jit
def instrength_jit(x):
    return jnp.sum(x, axis=0)

#tag strength
@jit
def strength_jit(x):
    return outstrength_jit(x) +  instrength_jit(x)

#tag outdegree
@jit
def outdegree_jit(x):
    mask = x != 0
    return jnp.sum(mask, axis=1)

#tag indegree
@jit
def indegree_jit(x):
    mask = x != 0
    return jnp.sum(mask, axis=0)

#tag degree
@jit
def degree_jit(x):
    return indegree_jit(x) +outdegree_jit(x)

#endregion <name> 

#region Class <comment>
class metrics:
    def __init__(self):
        pass
    @staticmethod 
    def eigen(m):
        return eigenvector_centrality(m)
    @staticmethod 
    def dijkstra(m,  source):
        return dijkstra(m, source)
    @staticmethod 
    def cc(m):
        return clustering_coefficient(m) 
    @staticmethod 
    def degree(m):
        return degree_jit(m)
    @staticmethod 
    def indegree(m):
        return indegree_jit(m)
    @staticmethod 
    def outdegree(m):
        return outdegree_jit(m)
    @staticmethod 
    def strength(m):
        return strength_jit(m)
    @staticmethod 
    def outstrength(m):
        return outstrength_jit(m)
    @staticmethod 
    def instrength(m):
        return instrength_jit(m)
#endregion <name>


