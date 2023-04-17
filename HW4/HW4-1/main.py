import numpy as np
import argparse
import matplotlib.pyplot as plt
from data_generator import data_generator
from logistic_regression import logistic_regression, predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--delta', type=float, default=None, help='if sum of square weight < delta, stop')
    parser.add_argument('--max_iter', type=int, default=None, help='maximum iteration')
    parser.add_argument('--N', type=int, default=50, help='Number of datapoint for one dataset')
    parser.add_argument('--mx1', type=float, default=1, help='mean of the first dataset\'s x coordinate')
    parser.add_argument('--vx1', type=float, default=2, help='variance of the first dataset\'s x coordinate')
    parser.add_argument('--my1', type=float, default=1, help='mean of the first dataset\'s y coordinate')
    parser.add_argument('--vy1', type=float, default=2, help='variance of the first dataset\'s y coordinate')
    parser.add_argument('--mx2', type=float, default=10, help='mean of the second dataset\'s x coordinate')
    parser.add_argument('--vx2', type=float, default=2, help='variance of the second dataset\'s x coordinate')
    parser.add_argument('--my2', type=float, default=10, help='mean of the second dataset\'s y coordinate')
    parser.add_argument('--vy2', type=float, default=2, help='variance of the second dataset\'s y coordinate')
    parser.add_argument('--plot_path', type=str, default='./plot.png', help='the path and filename to save the plot result')

    args = parser.parse_args()
    return args


def dataset():
    # (N, 2)
    D1 = data_generator(args.mx1, args.vx1, args.my1, args.vy1, args.N)
    D2 = data_generator(args.mx2, args.vx2, args.my2, args.vy2, args.N)
    # (2N, 2)
    D = np.concatenate([D1, D2])

    # design matrix, (2N, 3)
    ones = np.ones((D.shape[0], 1))
    A = np.hstack((ones, D))

    # ground truth, (2N, 1)
    y = np.zeros((D.shape[0], 1))
    y[D1.shape[0]:] = 1

    return A, y


def confusion_matrix(y, y_pred, verbose=True):
    cm = {}
    cm['TP'] = ((y == 0) & (y_pred == 0)).sum()
    cm['TN'] = ((y == 1) & (y_pred == 1)).sum()
    cm['FP'] = args.N - cm['TN']
    cm['FN'] = args.N - cm['TP']

    if verbose:
        print("Confusion Matrix:")
        print("             Predict cluster 1 Predict cluster 2")
        print(f"Is cluster 1      {cm['TP']}              {cm['FN']}")
        print(f"Is cluster 2      {cm['FP']}              {cm['TN']}")

    return cm


def sensitivity(cm):
    return cm['TP'] / (cm['TP'] + cm['FN'])


def specificity(cm):
    return cm['TN'] / (cm['FP'] + cm['TN'])


def print_result(W, y, y_pred):
    print("w: ")
    print(*[str(row)[1:-1] for row in W], sep='\n')
    print()
    cm = confusion_matrix(y, y_pred, verbose=True)
    print()
    print(f"Sensitivity (Successfully predict cluster 1): {sensitivity(cm)}")
    print(f"Specificity (Successfully predict cluster 2): {specificity(cm)}")
    print()
    print("-"*20)


def visualization(ax, A, y, title):
    point_x = A[:, 1]
    point_y = A[:, 2]
    idx_0 = (y == 0).squeeze(1)

    ax.scatter(point_x[idx_0], point_y[idx_0], color='red')
    ax.scatter(point_x[~idx_0], point_y[~idx_0], color='blue')
    ax.set_title(title)


if __name__ == '__main__':
    args = parse_args()

    # prepare data
    A, y = dataset()

    # initial weight
    W0 = np.random.rand(A.shape[1], 1)

    # -----------------------------Logistic Regression-----------------------------
    # Gradient Descent
    print("Gradient descent:\n")
    W_G = logistic_regression(W0, A, y, gd=True, lr=args.lr, delta=args.delta, max_iter=args.max_iter)
    y_pred_G = predict(W_G, A)
    print_result(W_G, y, y_pred_G)

    # Newton's method 
    print("Newton's method:\n")
    W_N = logistic_regression(W0, A, y, gd=False, lr=args.lr, delta=args.delta, max_iter=args.max_iter)
    y_pred_N = predict(W_N, A)
    print_result(W_N, y, y_pred_N)

    # visualization
    fig, axes = plt.subplots(1, 3)

    # plot Ground truth
    visualization(axes[0], A, y, 'Ground truth')
    # plot Gradient descent
    visualization(axes[1], A, y_pred_G, 'Gradient descent')
    # plot Newton's method
    visualization(axes[2], A, y_pred_N, 'Newton\'s method')

    fig.savefig(args.plot_path)
