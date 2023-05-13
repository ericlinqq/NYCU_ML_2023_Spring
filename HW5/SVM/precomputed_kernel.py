import numpy as np
from scipy.spatial.distance import cdist


def linearRBF(X, X_, gamma):
    linear = X @ X_.T
    RBF = np.exp(-gamma * cdist(X, X_, 'sqeuclidean'))
    kernel = linear + RBF
    kernel = np.hstack((np.arange(1, len(X)+1).reshape(-1, 1), kernel))

    return kernel