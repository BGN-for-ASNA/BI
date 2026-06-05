import numpy as np


def generate_gplvm_data(N=30, D=4, L=2, seed=42):
    rng = np.random.RandomState(seed)

    true_ls = np.array([1.0, 0.5])
    true_var = 1.5
    true_noise = 0.1

    X_true = rng.randn(N, L)

    diff = X_true[:, None, :] - X_true[None, :, :]
    sq_dist = np.sum((diff / true_ls) ** 2, axis=-1)
    K = true_var * np.exp(-0.5 * sq_dist) + (true_noise + 1e-6) * np.eye(N)
    K = (K + K.T) / 2

    Y = rng.multivariate_normal(np.zeros(N), K, size=D).T  # N x D

    X_prior_mean = np.zeros((N, L))

    true_params = {
        "log_lengthscale_0": float(np.log(true_ls[0])),
        "log_lengthscale_1": float(np.log(true_ls[1])),
        "log_variance": float(np.log(true_var)),
        "log_noise": float(np.log(true_noise)),
    }

    return Y, X_prior_mean, true_params, X_true
