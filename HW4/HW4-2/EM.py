import numpy as np
import os
from numba import jit, prange
from plot import plot_imagine
# from label_mapping import naive_matching
from label_mapping import perfect_matching, true_p


def binning(X):
    '''
    convert pixel to binary scale
    '''
    X = X.reshape(len(X), -1)
    X = (X > 127).astype(np.float32)
    return X


@jit
def log_bernoulli(Xn, pk):
    '''
    To compute log bernoulli
    Xn = D dimensional vector (a row of X)
    pk = D dimensional vector (a row of p)
    '''
    return (Xn * np.log(pk) + (1-Xn) * np.log(1-pk)).sum()


@jit
def E_step(X, w, lamb, p):
    '''
    To compute responsibility w
    X = N x D matrix (flatten images)
    lamb = K dimensional vector (probability of the given cluster is observed)
    p = K x D matrix (probability of a pixel to be 1 given a cluster)
    w = N x K matrix (responsibility)
    '''
    N = len(X)
    K = len(p)
    for n in prange(N):
        for k in prange(K):
            # to prevent underflow
            w[n, k] = np.log(lamb[k]) + log_bernoulli(X[n], p[k])

        # normalize
        # to prevent underflow
        w[n, :] -= w[n, :].max()
        w[n, :] = np.exp(w[n, :]) / np.exp(w[n, :]).sum()
        for k in prange(K):
            if w[n, k] < 1e-5:
                w[n, k] = 0

    return w


@jit
def M_step(X, w, lamb, p):
    '''
    Update the parameters using the current responsibility
    X = N x D matrix
    lamb = K dimensional vector
    p = K x D matrix
    w = N x K matrix
    '''
    N = len(X)
    D = X.shape[1]
    K = len(p)

    for k in prange(K):
        w_sum = w[:, k].sum()
        lamb[k] = w_sum

        p[k] = 0
        for n in prange(N):
            p[k] += w[n, k] * X[n]
        p[k] = (p[k] + 1e-13) / (w_sum + 1e-13 * D)

    lamb /= N
    # to prevent zero
    lamb[lamb < 1e-13] = 1e-13

    return lamb, p


@jit
def log_likelihood(X, w, lamb, p):
    '''
    To compute expectation of likelihood
    '''
    N = len(X)
    K = len(p)

    ll = 0
    for n in prange(N):
        for k in prange(K):
            ll += w[n, k] * (np.log(lamb[k]) + log_bernoulli(X[n], p[k]))

    return ll


def EM(X, y=None, n_cluster=10, max_iter=None, delta=None, verbose=False, record_path=None):
    if delta is None and max_iter is None:
        raise "delta and max_iter cannot both be None !!!!"

    if verbose and record_path is not None:
        f = open(os.path.join(record_path, 'record.txt'), 'w')

    X = binning(X)
    N = len(X)
    D = X.shape[1]
    K = n_cluster

    # initialization
    lamb = np.random.rand(K)
    lamb /= lamb.sum()
    p = np.random.rand(K, D)
    w = np.random.rand(N, K)

    if y is not None:
        gt_p = true_p(X, y)

    n_iter = 0
    ll_old = 0
    while max_iter is None or n_iter < max_iter:
        n_iter += 1
        print(f"iteration: {n_iter}", end='\r')

        # E step
        w = E_step(X, w, lamb, p)

        # M step
        lamb, p = M_step(X, w, lamb, p)

        # label mapping
        pred_cluster = np.argmax(w, axis=1)
        if y is not None:
            # mapping = label_mapping(pred_cluster, y)
            mapping = perfect_matching(p, gt_p)

        # plot imagination
            if verbose:
                plot_imagine(p, mapping, file=f)

        # calculate log likelihood
        ll = log_likelihood(X, w, lamb, p)

        # calculate difference
        difference = np.abs(ll - ll_old)

        if verbose:
            print(f"No. of Iteration: {n_iter}, Difference: {difference}", file=f)
            print("- " * 30, file=f)
            print(file=f)

        # check convergence
        if delta is not None and difference < delta:
            print("Converge!")
            break
        else:
            ll_old = ll

    if verbose and record_path is not None:
        f.close()

    return pred_cluster, mapping,  n_iter