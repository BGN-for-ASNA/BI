from BI.Utils.np_dists import UnifiedDist as dist
from BI.BNN.activations import activation

from numpyro import deterministic
import jax.numpy as jnp
import jax

class bnn(activation):
    """
    The bnn class is designed to build Bayesian Neural Networks (BNNs). It provides methods for creating network layers with specified prior distributions and activation functions. Additionally, it includes a specific two-layer BNN model for covariance estimation and a utility function to compute a correlation matrix from posterior samples.
    """
    def __init__(self):
        super().__init__()

        
        # Create the mapping in the constructor
        self._activation_map = {
            # Standard functions
            "relu": self.relu,
            "tanh": self.tanh,
            "sigmoid": self.sigmoid,
            "softmax": self.softmax,
            "linear": self.linear,

            # Advanced/Modern functions
            "leaky_relu": self.leaky_relu,
            "elu": self.elu,
            "gelu": self.gelu,
            "silu": self.silu,
            "swish": self.silu,  # Common alias for SiLU

            # Specialty functions
            "softplus": self.softplus,
        }

    def available_activations(self):
        """
        Returns a list of available activation functions.
        
        This method retrieves the names of all activation functions defined in the class.
        """
        return list(self._activation_map.keys())

    def layer_linear(self, X, dist, activation=None, bias=False):
        """        
        Adds a layer to the BNN with the specified prior distribution and activation function.       

        Parameters:
        - prior_dist (bi.dist): The prior distribution for the weights of the layer. The shape of the distribution defines the layer's input/output dimensions.
        - activation (str): The name of the activation function to use after this layer ('relu', 'tanh', 'sigmoid', 'softmax').  
        """

        if bias:
            prod = jnp.matmul(X, dist) + bias
        else:
            prod = jnp.matmul(X, dist) 

        # 2. Get and store the activation function object.
        if activation is None:
            return prod
        else:
            try:
                activation_func = getattr(self, activation)
            except AttributeError:
                raise ValueError(f"Unknown activation function: '{activation_name}'")    
            return activation_func(prod)

    def layer_attention(self, b_kv, b_q, d_model=32,  sample=True, name = ''):
        # Layers
        ### Dimensions 
        self.b_q, self.b_kv, self.d_model = b_q, b_kv, d_model
        ### Create learnable vector embeddings for each feature in the Query  block a
        self.emb_q = dist.normal(0,1,shape=(self.b_q,self.d_model), sample=sample, name=f'attention_q_{name}')   

        ### Create learnable vector embeddings for each feature in the Key/ Value block 
        self.emb_kv= dist.normal(0,1,shape=(self.b_kv,d_model),sample=sample, name=f'attention_kv_{name}')    

        ### Define linear layers to project embeddings into Query, Key, and Value spaces
        self.q_k_v_proj = dist.normal(0,1,shape=(self.d_model,self.d_model,3),sample=sample, name=f'attention_q_k_v_{name}')  

        ### Define a final output layer to map the attention context to the desired matrix shape.
        self.out = dist.normal(0,1,shape=(self.d_model * self.b_q, self.b_q * self.b_kv),  sample=sample, name=f'attention_out_{name}') 

        # Performs the attention mechanism
        ## three dense (linear) layers : Q = X W_Q,  K = X W_K, V = X W_V
        ## the three projection layers in attention do not have activation functions. They are purely linear transformations.
        Q = self.layer_linear(X = self.emb_q, dist = self.q_k_v_proj[:,:,0])
        K = self.layer_linear(X = self.emb_kv, dist = self.q_k_v_proj[:,:,1])
        V = self.layer_linear(X = self.emb_kv, dist = self.q_k_v_proj[:,:,2])    

        ## Attention mechanism: 
        ### Calculate dot-product similarity scores between all Queries and Keys. Scale for stability.
        scores =jnp.matmul(Q, K.T) / jnp.sqrt(self.d_model ** 0.5)

        #### Convert raw scores into attention weights (probabilities) using softmax.
        attn = jax.nn.softmax(scores, axis=-1)

        ####Compute the context vector as a weighted average of the Value vectors. Now we normalize the similarity scores across all possible `j` for each `i`
        context = jnp.matmul(attn, V)  

        # Reshape and pass through the final output layer to get the L_ij block.
        return self.layer_linear(context.reshape(1, -1), self.out).reshape(self.b_q, self.b_kv) 

    def scaled_dot_product_attention(self, Q, K, V):
        """Compute scaled dot-product attention."""
        d_k = Q.shape[-1]
        scores = jnp.matmul(Q, K.T) / jnp.sqrt(d_k)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(attn_weights, V), attn_weights

    def __call__(self, X):
        """Forward pass through the Bayesian attention mechanism."""
        Q = self.layer(X, self.q_proj)
        K = self.layer(X, self.k_proj)
        V = self.layer(X, self.v_proj)

        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)
        return attn_output, attn_weights
        
    def cov(self,hidden_dim,N,a, b, sample = True):
        """
        Creates a Bayesian Neural Network (BNN) with two layers for covariance estimation.
        The first layer maps the input to a hidden dimension using a normal distribution,
        and the second layer outputs two values per N (offsets for a and b).
        Parameters:
        - hidden_dim (int): The number of hidden units in the first layer.
        - N (int): The number of data points, which determines the size of the input and output.
        - a (jnp.ndarray): The first set of offsets for the covariance matrix.
        - b (jnp.ndarray): The second set of offsets for the covariance matrix.
        """
        # First layer weights/biases: note these are treated as latent parameters
        W1 = dist.normal(0, 1, shape=(N, hidden_dim), name='W1', sample=sample)

        # Second layer weights/biases
        W2 = dist.normal(0, 1, shape=(hidden_dim, 2), name='W2', sample=sample)

        # Create one-hot encoding for each N (each row is a one–hot vector)
        X = jnp.eye(N)

        hidden = jnp.tanh(jnp.dot(X, W1))  # shape: (N, hidden_dim)

        # Second layer: output two values per cafe (offsets for a and b)
        delta = jnp.dot(hidden, W2)        # shape: (N, 2)    

        return deterministic('rf', jnp.stack([a, b]) + delta) 

    def get_rho(self, posterior):
        """

        """
        a_b = jnp.mean(posterior, axis=0) 
        N= a_b.shape[0]

        # 1. Compute sample covariance matrix
        mean_a_b = jnp.mean(a_b, axis=0)  # Mean of [a_cafe, b_cafe]

        centered_data = a_b - mean_a_b  # Center data by subtracting the mean

        cov_sample = jnp.dot(centered_data.T, centered_data) / (N - 1)  # Covariance matrix

        # 2. Extract sigma (standard deviations) from the diagonal of the covariance matrix
        sigma = jnp.sqrt(jnp.diagonal(cov_sample))  # Extract standard deviations (sqrt of     variance)

        # 3. Compute Rho (correlation matrix)
        rho = cov_sample / (sigma[:, None] * sigma[None, :])  # Normalize covariance to obtain correlation matrix
        return rho

nn = bnn()
