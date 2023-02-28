from matrix import matrixTranspose, matrixMultiply, matrixInverse
import numpy as np

def LSE(A, b, lamb):
    n = A.shape[1]
    I = np.identity(n, dtype=float)
    At = matrixTranspose(A)
    G = matrixMultiply(At, A)
    x = matrixInverse(G + lamb * I)
    x = matrixMultiply(x, At)
    x = matrixMultiply(x, b)

    return x

def Newton(A, b):
    At = matrixTranspose(A)
    G = matrixMultiply(At, A)
    init = np.random.rand(A.shape[1], 1)
    H = 2 * G
    grad = matrixMultiply(H, init) - matrixMultiply(2 * At, b) 
    final = init - matrixMultiply(matrixInverse(H), grad)

    return final

def predict(A, x):
    return matrixMultiply(A, x)

def estimateError(pred, b):
    error = pred - b
    squared_error = matrixMultiply(matrixTranspose(error), error)
    return squared_error.item()