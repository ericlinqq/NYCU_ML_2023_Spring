from scipy.optimize import linear_sum_assignment
import numpy as np


def naive_matching(pred_cluster, true_label):
    '''
    mapping based on the number of the label in a cluster
    mapping: cluster -> class
    '''
    mapping = np.zeros(10) - 1

    for k in range(10):
        unique, counts = np.unique(true_label[np.where(pred_cluster == k)], return_counts=True)
        if len(unique) > 0:
            mapping[k] = unique[counts.argmax()]

    return mapping


def true_p(X, y):
    '''
    calculate the parameter p given y
    return gt_p = K x D matrix
    '''
    K = 10
    D = X.shape[1]

    gt_p = np.zeros((K, D))

    for k in range(K):
        gt_p[k] = np.sum(X[y == k], axis=0)
        gt_p[k] /= gt_p[k].sum()

    return gt_p


def perfect_matching(p, gt_p):
    '''
    using hungarian algorithm to match our estimation p to gt_p
    mapping: cluster -> class
    '''
    K = len(p)
    cost = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            cost[i, j] = np.linalg.norm(p[i] - gt_p[j])

    _, mapping = linear_sum_assignment(cost)

    return mapping