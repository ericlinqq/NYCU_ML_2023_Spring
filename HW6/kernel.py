import numpy as np
from scipy.spatial.distance import squareform, pdist


def user_defined_kernel(img, width, **kernel_param):
    gamma_s = kernel_param.get('gamma_s', 0.001)
    gamma_c = kernel_param.get('gamma_c', 0.001)

    # Compute S (coordinate of the pixel)
    n = len(img)
    S = np.zeros((n, 2))
    for i in range(n):
        S[i] = [i // width, i % width]

    K = squareform(np.exp(-gamma_s * pdist(S, 'sqeuclidean'))) * squareform(np.exp(-gamma_c * pdist(img, 'sqeuclidean')))

    return K