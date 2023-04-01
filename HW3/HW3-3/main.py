import argparse
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd.replace('HW3-3', 'HW3-1'))
from bayesian_linear_regression import bayesian_linear_regression, visualization
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./test.txt')
    parser.add_argument('--record_path', type=str, default='./result.txt')
    parser.add_argument('--plot_path', type=str, default='./plot.png')
    parser.add_argument('--size', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=None)
    parser.add_argument('--delta', type=float, default=None)
    args = parser.parse_args()

    return args


def load_data(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
        try:
            n = int(data[0].strip('\n'))
            a = float(data[1].strip('\n'))
            b = float(data[2].strip('\n'))
            W = data[3].strip('\n').split(' ')
            W = np.array(W, dtype=np.float64).reshape(-1, 1)
            assert len(W) == n
        except ValueError:
            print("invalid input")

    return n, a, b, W


if __name__ == '__main__':
    args = parse_args()
    print("Loading...")
    n, a, b, W = load_data(args.data_path)
    print("Calculating...")
    record_posterior, record_data = bayesian_linear_regression(n, a, b, W, args.size, args.max_iter, args.delta, args.record_path)
    print("Plotting...")
    visualization(W, a, record_posterior, record_data, filepath=args.plot_path)
