import numpy as np


def logistic_regression(W, A, y, mode='gd', lr=0.5, delta=None, max_iter=None):
    if mode != 'gd' and mode != 'newton':
        raise "mode should be either 'gd' or 'newton' !!!!"

    if delta is None and max_iter is None:
        raise "delta and max_iter cannot both be None !!!!"

    n_iter = 0
    while True:
        n_iter += 1
        if max_iter is not None and n_iter >= max_iter:
            break

        grad = gradient(W, A, y)

        if delta is not None and (grad**2).sum() < delta:
            break

        if mode == 'gd':
            W +=  lr * grad
        else:
            H = hessian(W, A, y)
            try:
                update = np.linalg.inv(H) @ grad
            except np.linalg.LinAlgError:
                update = grad

            W += lr * update

    return W


def predict(W, A):
    y_pred = logistic(A@W)
    idx_0 = (y_pred < 0.5)
    y_pred[idx_0] = 0
    y_pred[~idx_0] = 1

    return y_pred


def logistic(x):
    return 1 / (1+np.exp(-x))


def gradient(W, A, y):
    return A.T @ (y - logistic(A@W))


def hessian(W, A, y):
    D = np.identity(A.shape[0])

    eAW = np.exp(-A@W)
    # diag = eAW / (1+eAW)**2 <- overflow
    diag = 1/(1+eAW) - 1/(1+eAW)/(1+eAW)

    np.fill_diagonal(D, diag)

    return A.T @ D @ A
