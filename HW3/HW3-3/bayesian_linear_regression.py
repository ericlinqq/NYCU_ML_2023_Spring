import os
import sys
cwd = os.getcwd()
sys.path.append(cwd.replace('HW3-3', 'HW3-1'))
from data_generator import polynomial_basis_linear_generator, phi
import numpy as np
import copy
import matplotlib.pyplot as plt


def bayesian_linear_regression(n, a, b, W, size=1, max_iter=None, delta=None, filepath=None):
    if max_iter is None and delta is None:
        raise("max_iter and delta CANNOT both be None!!!!")

    if filepath is not None:
        f = open(filepath, 'w')

    data_num = 0
    prior = {"mean": np.zeros((n, size)), "precision": b * np.identity(n)}
    last_prior = {"mean": None, "precision": None}
    predictive = {"mean": None, "precision": None}
    record_posterior = {"10": None, "50": None, "final": None}
    record_data = {"X": [], "y": []}

    while True:
        if max_iter is not None and data_num >= max_iter:
            break

        # new datapoint
        X, y = polynomial_basis_linear_generator(n, a, W, size, return_X=True)
        data_num += 1

        record_data['X'].append(X)
        record_data['y'].append(y)

        # calculate posterior distribution
        last_prior = copy.deepcopy(prior)
        prior['mean'], prior['precision'] = posterior_distribution(X, y, a, last_prior['mean'], last_prior['precision'])

        if data_num == 10:
            record_posterior['10'] = copy.deepcopy(prior)
        elif data_num == 50:
            record_posterior['50'] = copy.deepcopy(prior)

        if delta is not None and \
                (np.absolute(prior['mean'] - last_prior['mean']) < delta).all():
            break

        # calculate predictive distribution
        predictive['mean'], predictive['var'] = predictive_distribution(X, a, prior['mean'], prior['precision'])

        if filepath is not None:
            record_to_txt(X, y, prior, predictive, file=f)


    if filepath is not None:
        f.close()

    record_posterior['final'] = copy.deepcopy(prior)
    record_data['X'] = np.array(record_data['X'])
    record_data['y'] = np.array(record_data['y'])

    return record_posterior, record_data


def posterior_distribution(X, y, a, prior_mean, prior_precision):
    A = phi(prior_mean.shape[0], X)

    # General form
    posterior_precision = a * A.T @ A + prior_precision
    posterior_mean = np.linalg.inv(posterior_precision) @ (a * A.T @ y + prior_precision @ prior_mean)

    return posterior_mean, posterior_precision


def predictive_distribution(X, a, posterior_mean, posterior_precision):
    A = phi(posterior_mean.shape[0], X)

    predictive_mean = A @ posterior_mean
    predictive_var = []
    for ai in A:
        predictive_var.append(1./a + ai @ np.linalg.inv(posterior_precision) @ ai.T)

    return predictive_mean.flatten(), np.array(predictive_var)


def record_to_txt(X, y, posterior, predictive, file):
    print(f"Add data point ({X}, {y}):\n", file=file)

    print("Posterior mean:", file=file)
    print(posterior['mean'], file=file)
    print(file=file)

    print("Poseterior variance:", file=file)
    print(np.linalg.inv(posterior['precision']), file=file)
    print(file=file)

    print(f"Predictive distribution ~ N({predictive['mean']}, {predictive['var']})", file=file)
    print("_" * 80, file=file)


def visualization(W, a, record_posterior, record_data, ylim=(-20, 25), filepath=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    X = np.arange(-2, 2, 0.05)

    # plot ground truth
    y = fitting_line(W, X)
    axes[0, 0].plot(X, y, color='black')
    axes[0, 0].plot(X, y+a, color='red')
    axes[0, 0].plot(X, y-a, color='red')
    axes[0, 0].set_title('Ground truth')

    # plot predict result
    def plot_predict(ax, posterior, data_num, plot_title):
        mean, var = predictive_distribution(X, a, posterior['mean'], posterior['precision'])
        y_pred = fitting_line(posterior['mean'], X)
        ax.plot(X, y_pred, color='black')
        ax.plot(X, mean+var, color='red')
        ax.plot(X, mean-var, color='red')
        ax.scatter(record_data['X'][:data_num], record_data['y'][:data_num])
        ax.set_title(plot_title)

    plot_predict(axes[0, 1], record_posterior['final'], -1, 'Predict result')
    plot_predict(axes[1, 0], record_posterior['10'], 10, 'After 10 incomes')
    plot_predict(axes[1, 1], record_posterior['50'], 50, 'After 50 incomes')

    # set ylim
    for ax in axes.flatten():
        ax.set_ylim(*ylim)

    fig.savefig(filepath)


def fitting_line(W, X):
    return phi(W.shape[0], X) @ W
