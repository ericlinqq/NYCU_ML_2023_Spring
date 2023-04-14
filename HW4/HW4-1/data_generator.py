import numpy as np


def data_generator(mx, vx, my, vy, N):
    D = np.zeros((N, 2))
    X = normal_generator(mx, vx, size=N)
    y = normal_generator(my, vy, size=N)

    D[:, 0] = X
    D[:, 1] = y

    return D


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
