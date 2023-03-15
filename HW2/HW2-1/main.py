import argparse
import os
import numpy as np
from load_data import load_image, load_label
from naive_bayes import NaiveBayes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./', help='path of the directory contains the file we need')
    parser.add_argument('--record_path', default='./', help='path of the directory to store record file')
    parser.add_argument('--discrete', type=bool, default=0, help='discrete or continuous')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    X_train = load_image(os.path.join(args.data_path, 'train-images-idx3-ubyte'))
    y_train = load_label(os.path.join(args.data_path, 'train-labels-idx1-ubyte'))
    X_test = load_image(os.path.join(args.data_path, 't10k-images-idx3-ubyte'))
    y_test = load_label(os.path.join(args.data_path, 't10k-labels-idx1-ubyte'))

    nb = NaiveBayes(args.discrete)
    nb.fit(X_train, y_train)
    nb.record_to_txt_file(X_test, y_test, args.record_path)
