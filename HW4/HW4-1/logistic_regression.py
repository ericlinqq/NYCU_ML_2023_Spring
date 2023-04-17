import numpy as np


def logistic_regression(W, A, y, gd=True, lr=0.5, delta=None, max_iter=None):
    W_c = W.copy()

    if delta is None and max_iter is None:
        raise "delta and max_iter cannot both be None !!!!"

    n_iter = 0
    while True:
        n_iter += 1
        if max_iter is not None and n_iter >= max_iter:
            break

        grad = gradient(W_c, A, y)

        if delta is not None and (grad**2).sum() < delta:
            break

        update = grad
        if not gd:
            H = hessian(W_c, A, y)
            try:
                update = np.linalg.inv(H) @ grad
            except:
                print("Hessain matrix is singular !!!!")

        W_c += lr * update

    return W_c


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
