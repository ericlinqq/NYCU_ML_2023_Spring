import numpy as np
import argparse
from data_generator import normal_generator, polynomial_basis_linear_generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=0,
                        help='0: univariate gaussian data generator\n \
                              1: polymial basis linear model data generator')
    parser.add_argument('--mean', type=float, default=0, help='mean for univariate gaussian data generator')
    parser.add_argument('--var', type=float, default=1, help='variance for univariate gaussain data generator')
    parser.add_argument('--filepath', type=str, default='./test.txt',
                        help='testfile for polynomial basis linear model data generator')
    parser.add_argument('--size', type=int, default=1, help='output size')
    parser.add_argument('--return_x', action='store_true',
                        help='Whether to return X for polynomial basis linear model generator')
    args = parser.parse_args()
    return args


def load_data(filepath): # for polynomial basis linear model data generator
    with open(filepath, 'r') as f:
        data = f.readlines()
        try:
            n = int(data[0].strip('\n'))
            a = float(data[1].strip('\n'))
            W = data[2].strip('\n').split(' ')
            W = np.array(W, dtype=np.float64).reshape(-1, 1)
            assert len(W) == n
        except ValueError:
            print("invalid input")

    return n, a, W


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 0:  # univariate gaussian data generator
        print(normal_generator(args.mean, args.var, args.size))

    elif args.mode == 1:  # polynomial basis linear model data generator
        n, a, W = load_data(args.filepath)
        print(polynomial_basis_linear_generator(n, a, W, args.size, args.return_x))

    else:
        raise("--mode can only be either 0 or 1")
