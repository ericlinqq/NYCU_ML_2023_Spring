import numpy as np

def matrixTranspose(mat):
    m = mat.shape[0]
    n = mat.shape[1]

    matT = np.zeros([n, m], dtype=float)
    for i in range(m):
        for j in range(n):
            matT[j, i] = mat[i, j]
         
    return matT

def dot_product(vec1, vec2):
    sum = 0.0
    for i in range(len(vec1)):
        sum += vec1[i] * vec2[i]
    
    return sum

def matrixMultiply(mat1, mat2):
    m = mat1.shape[0]
    n = mat1.shape[1]
    r = mat2.shape[1]

    assert(n == mat2.shape[0])

    mat3 = np.zeros([m, r], dtype=float)
    for i in range(m):
        for k in range(r):
            mat3[i, k] = dot_product(mat1[i, :], mat2[:, k])
    
    return mat3

# input: a square matrix
# output: matrix [L\U]
def LUdecomp(mat):
    n = len(mat)

    for j in range(n - 1):
        for i in range(j + 1, n):
            if mat[i, j] != 0.0:
                fac = mat[i, j] / mat[j, j]
                mat[i, j+1:] -= fac * mat[j, j+1:]    # upper triangular matrix
                mat[i, j] = fac     # lower triangular matrix

    return mat

# input: matrix [L\U]
# output: vector x 
# LUx = b
# let y = Ux
# so Ly = b, solved y by forward substitution
# then Ux = y, solve x by backward substitution
def LUsolve(mat, b):
    n = len(mat)

    # forward substitution
    for i in range(1, n):
        b[i] -= dot_product(mat[i, :i], b[:i])

    # backward substitution
    b[n-1] /= mat[n-1, n-1]
    for i in range(n-2, -1, -1):
        b[i] = (b[i] - dot_product(mat[i, i+1:], b[i+1:])) / mat[i, i]
    
    return b

def matrixInverse(mat):
    n = len(mat) 
    mat = LUdecomp(mat)
    I = np.identity(n, dtype=float)
    mat_inverse = np.zeros([n, n], dtype=float)

    for i in range(n):
        mat_inverse[:, i] = LUsolve(mat, I[:, i])

    return mat_inverse 