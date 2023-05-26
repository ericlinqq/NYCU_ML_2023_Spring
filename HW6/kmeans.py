import numpy as np
from kernel import user_defined_kernel


class Kmeans:
    def __init__(self):
        np.random.seed(19990410)
        self.colormap = np.random.randint(0, 256, size=(100, 3))
        self.n_cluster = None
        self.centroid_method = None
        self.EPS = None


    def _euclidean(self, point, data):
        return np.sqrt(np.sum((point - data) ** 2, axis=1))


    def _initialize_centroid(self, X):
        centroids = np.zeros((self.n_cluster, X.shape[1]))
        if self.centroid_method == 'kmeans++':
            # randomly pick one point as the first centroid
            centroids[0] = X[np.random.choice(range(len(X)), size=1), :]

            for c in range(1, self.n_cluster):
                temp_dist = np.zeros((len(X), c))
                # compute distance
                for j in range(c):
                    temp_dist[:, j] = self._euclidean(centroids[j], X)
                # for each data point, select the minimum distance (nearest centroid)
                dists = np.min(temp_dist, axis=1)
                # calulate probability
                dists /= np.sum(dists)
                # roulette wheel selection
                new_centroid_idx = np.random.choice(range(len(X)), size=1, p=dists)
                centroids[c] = X[new_centroid_idx]

        elif self.centroid_method == 'random':
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            for c in range(X.shape[1]):
                centroids[:, c] = np.random.normal(X_mean[c], X_std[c], size=self.n_cluster)
        else:
            raise Exception ('unavailable centroid method!')

        return centroids


    def _cluster_color(self, assigned_cluster):
        color = self.colormap[:self.n_cluster, :]
        res = np.zeros((len(assigned_cluster), 3), dtype=np.uint8)
        for i in range(len(res)):
            res[i, :] = color[assigned_cluster[i]]

        return res


    def _E_step(self, mean, X):
        euclid_dist = np.zeros((len(X), self.n_cluster))
        for c in range(self.n_cluster):
            euclid_dist[:, c] = self._euclidean(mean[c], X)
        assigned_cluster = np.argmin(euclid_dist, axis=1)

        return assigned_cluster


    def _M_step(self, mean, assigned_cluster, X):
        new_mean = np.zeros(mean.shape)
        for i in range(self.n_cluster):
            assign_i = np.argwhere(assigned_cluster == i).reshape(-1)
            for j in assign_i:
                new_mean[i] += X[j]
            if len(assign_i) > 0:
                new_mean[i] /= len(assign_i)

        return new_mean


    def _kmeans(self, X, visualize=True):
        print("initialize centroid")
        init_mean = self._initialize_centroid(X)
        assigned_cluster = np.zeros(len(X), dtype=np.uint8)
        assigned_color = []
        n_iter = 1
        print("start EM")
        while True:
            print("E step")
            # E step
            assigned_cluster = self._E_step(init_mean, X)

            print("M step")
            # M step
            new_mean = self._M_step(init_mean, assigned_cluster, X)

            diff = np.sum((new_mean - init_mean) ** 2)
            init_mean = new_mean
            if visualize:
                assign_color = self._cluster_color(assigned_cluster)
                assigned_color.append(assign_color)

            print(f"Iteration {n_iter}")
            for i in range(self.n_cluster):
                print(f"cluster={i+1} : (N pixel assigned : {np.count_nonzero(assigned_cluster == i)})")
            print(f"Difference : {diff}")
            print()

            n_iter += 1
            if diff < self.EPS:
                self.centroids = new_mean
                break

        if visualize:
            return assigned_cluster, np.array(assigned_color)

        return assigned_cluster


    def fit(self, X, n_cluster=2, centroid_method='kmeans++', EPS=1e-9, visualize=False):
        if len(X.shape) != 2:
            raise Exception ('Input shoud be a 2d matrix!')
        self.n_cluster = n_cluster
        self.centroid_method = centroid_method
        self.EPS = EPS

        print("starting k means")
        return self._kmeans(X, visualize=visualize)


    def evaluate(self, X_test):
        return self._E_step(self.centroids, X_test)
