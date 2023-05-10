import numpy as np
import argparse
import os
from dataloader import load_data
from GaussianProcess import GaussianProcess
from kernel import rational_quadratic_kernel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/', type=str, help="input data path")
    parser.add_argument('--fig_path', default='./visualization/', type=str, help="output figure path")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.data_path):
        raise "input data path does not exist !!!!"

    if not os.path.exists(args.fig_path):
        os.makedirs(args.fig_path)

    X_train, y_train = load_data(args.data_path)
    X_test = np.linspace(-60, 60, num=600)
    GP = GaussianProcess(rational_quadratic_kernel, beta=5)
    mean, var = GP.fit(X_train).predict(X_test, y_train)
    GP.visualization(title="length_scale=1.0, alpha=1.0", fig_path=args.fig_path, fig_name='original.png')

    mean_opt, var_opt = GP.optimize_kernel_param(X_train, y_train,
                            bounds=((1e-6, None), (1e-6, None)),
                            length_scale=1, alpha=1) \
        .predict(X_test, y_train)
    GP.visualization(title=f"length_scale={GP.kernel_param['length_scale']}, alpha={GP.kernel_param['alpha']}",
                        fig_path=args.fig_path,
                        fig_name='optimized.png')
