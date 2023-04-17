import argparse
import os
from load_data import load_image, load_label
from EM import EM
from metric import confusion_matrices, accuracy, print_metric
from plot import plot_imagine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../HW2/HW2-1/', help='')
    parser.add_argument('--max_iter', type=int, default=None, help='')
    parser.add_argument('--delta', type=float, default=None, help='')
    parser.add_argument('--record_path', type=str, default='./', help='')
    parser.add_argument('--metric_path', type=str, default='./', help='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    X = load_image(os.path.join(args.data_path, 'train-images-idx3-ubyte'))
    y = load_label(os.path.join(args.data_path, 'train-labels-idx1-ubyte'))

    pred_cluster, mapping, converge_iter = EM(X, y, max_iter=args.max_iter, delta=args.delta, verbose=True, record_path=args.record_path)
    error_rate = 1 - accuracy(pred_cluster, y, mapping)
    pred_label = mapping[pred_cluster]
    CM = confusion_matrices(pred_label, y)
    print_metric(CM, converge_iter, error_rate, metric_path=args.metric_path)