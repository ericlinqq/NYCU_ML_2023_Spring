import pandas as pd
import os


def load_data(data_path):
    X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"), header=None).to_numpy()
    y_train = pd.read_csv(os.path.join(data_path, "Y_train.csv"), header=None).to_numpy().reshape(-1)
    X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"), header=None).to_numpy()
    y_test = pd.read_csv(os.path.join(data_path, "Y_test.csv"), header=None).to_numpy().reshape(-1)

    return X_train, y_train, X_test, y_test