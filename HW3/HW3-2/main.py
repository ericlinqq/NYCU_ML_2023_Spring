import argparse
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd.replace('HW3-2', 'HW3-1'))
from data_generator import normal_generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='./result.txt', help='The filepath of the record file')
    parser.add_argument('--mean', type=float, default=0.0, help='The mean of the normal data generator')
    parser.add_argument('--var', type=float, default=1.0, help='The variance of the normal data generator')
    parser.add_argument('--max_iter', type=int, default=None, help='Maximum iteration of sequential estimate')
    parser.add_argument('--delta', type=float, default=None, help='Minimum difference between two iterations')
    args = parser.parse_args()

    return args


def sequential_estimator(filepath, mean, var, max_iter=None, delta=None):
    if max_iter is None and delta is None:
        raise("max_iter and delta CANNOT both be None!!!!")

    f = open(filepath, 'w')
    print(f"Data point source function: N({mean}, {var})", file=f)
    print(file=f)

    data_num = 0
    sum_data = 0.
    sum_square_data = 0.
    last_record = {"mean": 0, "var": 0}

    while True:
        if max_iter is not None and data_num >= max_iter:
            break

        new_data = normal_generator(mean, var, 1).item()
        data_num += 1
        sum_data += new_data
        sum_square_data += new_data * new_data
        estimate_mean = sum_data / data_num
        estimate_var = (sum_square_data - sum_data*sum_data/data_num) / (data_num-1 + 1e-15)

        if delta is not None and \
            abs(last_record['mean'] - estimate_mean) < delta and \
                abs(last_record['var'] - estimate_var) < delta:
            break

        last_record['mean'] = estimate_mean
        last_record['var'] = estimate_var

        print(f"Add data point: {new_data}", file=f)
        print(f"Mean = {estimate_mean}  Variance = {estimate_var}", file=f)

    f.close()

if __name__ == '__main__':
    args = parse_args()
    sequential_estimator(args.filepath, args.mean, args.var, args.max_iter, args.delta)
