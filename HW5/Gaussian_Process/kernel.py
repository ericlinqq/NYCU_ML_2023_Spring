import numpy as np


def rational_quadratic_kernel(x_a, x_b, **kernel_param):
    length_scale = kernel_param.get('length_scale', 1.0)
    alpha = kernel_param.get('alpha', 1.0)
    SE = np.power(x_a.reshape(-1, 1) - x_b.reshape(1, -1), 2)

    return np.power(1 + SE / (2 * alpha * length_scale), -alpha)
