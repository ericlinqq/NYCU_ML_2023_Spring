import numpy as np


def normal_generator(mean=0.0, var=1.0, size=1):
    """
    Using Marsagalia polar method
    """
    U = np.random.uniform(-1, 1, size)
    V = np.random.uniform(-1, 1, size)
    S = U*U + V*V

    while (S >= 1).any():
        replace_indices = (S >= 1)
        replace_size = replace_indices.sum()
        U[replace_indices] = np.random.uniform(-1, 1, replace_size)
        V[replace_indices] = np.random.uniform(-1, 1, replace_size)
        S[replace_indices] = U[replace_indices]*U[replace_indices] + V[replace_indices]*V[replace_indices]

    Z = U * np.sqrt(-2. * np.log(S) / S)

    return Z * np.sqrt(var) + mean


def phi(n, X):
    A = np.zeros((len(X), n), dtype=np.float64)
    for i in range(n):
        A[:, i] = np.power(X, i)
    return A


def polynomial_basis_linear_generator(n, a, W, size=1, return_X=False):
    X = np.random.uniform(-1, 1, size).reshape(-1, 1)
    e = normal_generator(0, a, size).reshape(-1, 1)
    A = phi(n, X)

    if return_X:
        return X, A @ W + e

    return A @ W + e
