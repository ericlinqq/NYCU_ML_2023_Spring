import numpy as np
import argparse
from data_generator import data_generator
from logistic_regression import logistic_regression, predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.05, help='')
    parser.add_argument('--delta', type=float, default=None, help='')
    parser.add_argument('--max_iter', type=int, default=None, help='')
    parser.add_argument('--N', type=int, default=50, help='')
    parser.add_argument('--mx1', type=float, default=1, help='')
    parser.add_argument('--my1', type=float, default=1, help='')
    parser.add_argument('--mx2', type=float, default=10, help='')
    parser.add_argument('--my2', type=float, default=2, help='')
    parser.add_argument('--vx1', type=float, default=2, help='')
    parser.add_argument('--vy1', type=float, default=2, help='')
    parser.add_argument('--vx2', type=float, default=2, help='')
    parser.add_argument('--vy2', type=float, default=2, help='')

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
    cm['TP'] = ((y==0) == (y_pred==0)).sum()
    cm['TN'] = ((y==1) == (y_pred==1)).sum()
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
    print(W)
    print()
    cm = confusion_matrix(y, y_pred, verbose=True)
    print()
    print(f"Sensitivity (Successfully predict cluster 1): {sensitivity(cm)}")
    print(f"Specificity (Successfully predict cluster 2): {specificity(cm)}")
    print()
    print("-"*20)


if __name__ == '__main__':
    args = parse_args()

    # prepare data
    A, y = dataset()

    # initial weight
    W0 = np.random.rand(A.shape[1], 1)

    # -----------------------------Logistic Regression-----------------------------
    # Gradient Descent
    W_G = logistic_regression(W0, A, y, 'gd', args.lr, args.delta, args.max_iter)
    y_pred_G = predict(W_G, A)
    print("Gradient descent:\n")
    print_result(W_G, y, y_pred_G)

    # Newton's method
    W_N = logistic_regression(W0, A, y, 'newton', args.lr, args.delta, args.max_iter)
    y_pred_N = predict(W_N, A)
    print("Newton's method:\n")
    print_result(W_N, y, y_pred_N)
