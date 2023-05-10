import numpy as np
import os


def load_data(data_path):
    X = []
    y = []
    with open(os.path.join(data_path, 'input.data'), 'r') as f:
        for line in f.readlines():
            line = line.split()
            X.append(float(line[0]))
            y.append(float(line[1]))

    return np.array(X), np.array(y)
