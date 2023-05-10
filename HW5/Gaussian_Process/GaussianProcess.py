import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize


class GaussianProcess():
    def __init__(self, kernel_func, *, beta=5, **kernel_param):
        self.kernel_func = kernel_func
        self.kernel_param = kernel_param
        self.beta = beta
        self.X = None
        self.y = None
        self.C = None
        self.x_star = None
        self.mean = None
        self.var = None
        self.std = None

    def _covariance(self, X):
        return self.kernel_func(X, X, **self.kernel_param) + 1 / self.beta * np.identity(len(X))

    def fit(self, X):
        self.X = X
        self.C = self._covariance(self.X)

        return self

    def predict(self, x_star, y):
        self.y = y
        self.x_star = x_star
        k_x_s = self.kernel_func(self.X, x_star, **self.kernel_param)
        k_star = self.kernel_func(x_star, x_star, **self.kernel_param) + 1 / self.beta * np.identity(len(x_star))
        temp = k_x_s.T @ np.linalg.inv(self.C)
        self.mean = temp @ self.y
        self.var = k_star - temp @ k_x_s
        self.std = np.sqrt(np.diag(self.var))

        return self.mean, self.var

    def visualization(self, title, fig_path, fig_name='figure.png'):
        plt.figure(figsize=(20, 5))
        plt.plot(self.x_star, self.mean, color='lightseagreen', label='mean')
        plt.fill_between(self.x_star, self.mean + 2 * self.std, self.mean - 2 * self.std, facecolor='aquamarine', label='95% confidence interval')
        plt.scatter(self.X, self.y, color='mediumvioletred', label='training data')
        plt.legend(loc='upper right')
        plt.title(title)
        plt.savefig(os.path.join(fig_path, fig_name))

    def _objFunc(self, const_args):
        X, y, beta = const_args

        def negLogLikeLihood(x0):
            for idx, key in enumerate(self.kernel_param.keys()):
                self.kernel_param[key] = x0[idx]
            self.C = self._covariance(X)
            return 0.5 * np.log(np.linalg.det(self.C)) + 0.5 * y.T @ np.linalg.inv(self.C) @ y + len(X) / 2 * np.log(2 * np.pi)

        return negLogLikeLihood

    def optimize_kernel_param(self, X, y, *, bounds=None, **init_param):
        self.X = X
        self.y = y
        self.kernel_param = init_param
        const_args = (self.X, self.y, self.beta)
        x0 = tuple(init_param.values())

        res = minimize(self._objFunc(const_args), x0, bounds=bounds)
        for idx, key in enumerate(self.kernel_param.keys()):
            self.kernel_param[key] = res.x[idx]

        return self
