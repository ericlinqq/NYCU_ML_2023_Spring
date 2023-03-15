import numpy as np
import os


class NaiveBayes():
    def __init__(self, discrete=True):
        self.discrete = discrete
        if self.discrete:
            print("discrete")
        else:
            print("continuous")

    def _fit_discrete(self, X, y):
        X = X.astype(np.uint8)
        X = X / 8
        X = X.reshape(len(X), -1)

        self._label, self._prior = self._calculate_prior(y)
        self._likelihood = np.zeros((X.shape[1], 32, len(self._label)), dtype=np.float64)

        for idx_label, label in enumerate(self._label):
            X_match = X[y == label]
            for idx_pixel in range(X_match.shape[1]):
                X_hist = np.histogram(X_match[:, idx_pixel], bins=np.arange(33))[0]
                X_hist = X_hist / len(X_match)
                self._likelihood[idx_pixel, :, idx_label] = X_hist.T

        return self

    def _calculate_prior(self, y):
        label, counts = np.unique(y, return_counts=True)
        prior = counts / np.sum(counts)

        return label, prior

    def _predict_discrete(self, X, eps=1e-15):
        X = X.astype(np.uint8)
        X = X / 8
        X = X.reshape(len(X), -1)
        self._posterior = np.zeros((len(X), len(self._label)), dtype=np.float64)

        for num in range(len(X)):
            for idx_label, label in enumerate(self._label):
                likelihood = 0.
                for idx_pixel, pixel in enumerate(X[num]):
                    likelihood += np.log(self._likelihood[idx_pixel, pixel, idx_label] + eps)
                self._posterior[num, idx_label] = likelihood + np.log(self._prior[idx_label])

        self._posterior = self._posterior / np.sum(self._posterior, axis=1, keepdims=True)
        self._y_pred = self._label[np.argmin(self._posterior, axis=1)]

        return self._y_pred

    def _fit_continuous(self, X, y):
        X = X.astype(np.float64)
        X = X.reshape(len(X), -1)

        self._label, self._prior = self._calculate_prior(y)
        self._mean = np.zeros((len(self._label), X.shape[1]), dtype=np.float64)
        self._var = np.zeros((len(self._label), X.shape[1]), dtype=np.float64)

        for idx_label, label in enumerate(self._label):
            X_match = X[y == label]
            self._mean[idx_label, :] = np.mean(X_match, axis=0)
            self._var[idx_label, :] = np.var(X_match, axis=0)

        return self

    def _predict_continuous(self, X, eps=1e-15):
        X = X.astype(np.float64)
        X = X.reshape(len(X), -1)

        self._posterior = np.zeros((len(X), len(self._label)), dtype=np.float64)

        for num in range(len(X)):
            for idx_label, label in enumerate(self._label):
                likelihood = np.sum(np.log(self._gaussian_pdf(idx_label, X[num]) + eps))
                self._posterior[num, idx_label] = likelihood + np.log(self._prior[idx_label])

        self._posterior = self._posterior / np.sum(self._posterior, axis=1, keepdims=True)
        self._y_pred = self._label[np.argmin(self._posterior, axis=1)]

        return self._y_pred

    def _gaussian_pdf(self, idx_label, xi, eps=1e3):
        mean = self._mean[idx_label]
        var = self._var[idx_label] + eps
        denominator = np.sqrt(2. * var * np.pi)
        numerator = np.exp(-(np.square(xi - mean)) / (2. * var))

        return numerator / denominator

    def fit(self, X, y):
        print("fitting...")
        if self.discrete:
            return self._fit_discrete(X, y)
        return self._fit_continuous(X, y)

    def predict(self, X):
        print("predicting...")
        if self.discrete:
            return self._predict_discrete(X)
        return self._predict_continuous(X)

    def record_to_txt_file(self, X, y, path):
        self.predict(X)

        print("recording...")
        if self.discrete:
            filename = 'result_discrete.txt'
        else:
            filename = 'result_continuous.txt'

        with open(os.path.join(path, filename), 'w') as f:
            for num in range(len(self._y_pred)):
                print("Posterior (in log scale):", file=f)
                for i in range(10):
                    print(f"{i}: {self._posterior[num, np.argwhere(self._label == i).item()].item()}", file=f)
                print(f"Prediction: {self._y_pred[num]}, Ans: {y[num]}", file=f)
                print(file=f)

            print("Imagination of numbers in Bayesian classifier:", file=f)
            print(file=f)

            if self.discrete:
                for i in range(10):
                    print(f"{i}:", file=f)
                    idx_label = np.argwhere(self._label == i).item()
                    black = self._likelihood[:, :-16, idx_label].sum(axis=1)
                    white = self._likelihood[:, -16:, idx_label].sum(axis=1)
                    image = (white > black).reshape(28, 28).astype(np.uint8)
                    for row in range(image.shape[0]):
                        for col in range(image.shape[1]):
                            print(f"{image[row, col]} ", end='', file=f)
                        print(file=f)
                    print(file=f)
            else:
                for i in range(10):
                    print(f"{i}:", file=f)
                    idx_label = np.argwhere(self._label == i).item()
                    num_feat = self._mean.shape[1]
                    white = np.zeros((num_feat, ))
                    black = np.zeros((num_feat, ))
                    for i in range(128):
                        black = black + self._gaussian_pdf(idx_label, np.ones(num_feat)*i)
                    for i in range(128, 256):
                        white = white + self._gaussian_pdf(idx_label, np.ones(num_feat)*i)
                    image = (white > black).reshape(28, 28).astype(np.uint8)
                    for row in range(image.shape[0]):
                        for col in range(image.shape[1]):
                            print(f"{image[row, col]} ", end='', file=f)
                        print(file=f)
                    print(file=f)

            acc = self._calculate_accuracy(self._y_pred, y)
            print(f"Error rate: {1. - acc}", file=f)

    def _calculate_accuracy(self, y_pred, y):
        return np.sum(y_pred == y) / len(y)
