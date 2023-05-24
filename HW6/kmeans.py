import numpy as np
from kernel import user_defined_kernel


class Kmeans:
    def __init__(self, *, kernel_function=None, **kernel_param): 
        self.X = None
        self.height = None
        self.width = None
        self.kernel_function = kernel_function
        self.kernel_param = kernel_param
        np.random.seed(19990410)
        self.colormap = np.random.choice(range(256), size=(100, 3))
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

        return res.reshape(self.height, self.width, 3)

    
    def _kmeans(self, X):
        print("initialize centroid")
        init_mean = self._initialize_centroid(X)
        assigned_cluster = np.zeros(len(X), dtype=np.uint8)
        segment = []
        n_iter = 1
        print("start EM")
        while True:
            print("E step")
            # E step
            euclid_dist = np.zeros((len(X), self.n_cluster))
            for c in range(self.n_cluster):
                euclid_dist[:, c] = (self._euclidean(init_mean[c], X))
            assigned_cluster = np.argmin(euclid_dist, axis=1)

            print("M step")
            # M step
            new_mean = np.zeros(init_mean.shape)
            for i in range(self.n_cluster):
                assign_i = np.argwhere(assigned_cluster == i).reshape(-1)
                for j in assign_i:
                    new_mean[i] += X[j]
                if len(assign_i) > 0:
                    new_mean[i] /= len(assign_i)

            diff = np.sum((new_mean - init_mean) ** 2)
            init_mean = new_mean
            assign_color = self._cluster_color(assigned_cluster)
            segment.append(assign_color)

            print(f"Iteration {n_iter}")
            for i in range(self.n_cluster):
                print(f"cluster={i+1} : (N pixel assigned : {np.count_nonzero(assigned_cluster == i)})")
            print(f"Difference : {diff}")
            print()
            
            n_iter += 1
            if diff < self.EPS:
                self.centroids = new_mean
                break

        return assigned_cluster, segment


    def fit(self, X, n_cluster=2, centroid_method='kmeans++', EPS=1e-9):
        if len(X.shape) == 3:
            self.height = X.shape[0]
            self.width = X.shape[1]
            self.X = X.reshape(-1, X.shape[-1])
        else:
            self.X = X
            self.height = 100
            self.width = 100
        self.n_cluster = n_cluster
        self.centroid_method = centroid_method
        self.EPS = EPS
        if self.kernel_function == None:
            return self._kmeans(self.X)
        elif self.kernel_function == user_defined_kernel: 
            print("compute kernel function")
            gram = self.kernel_function(self.X, self.width, **self.kernel_param)
        else:
            gram = self.kernel_function(self.X, **self.kernel_param)
        print("starting k means")
        return self._kmeans(gram)


    def evaluate(self, X_test):
        assigned_cluster = np.zeros(len(X_test), dtype=np.uint8)
        for i in range(len(X_test)):
            euclid_dist = []
            for j in range(self.n_cluster):
                euclid_dist.append(self._euclidean(self.centroids[j], X_test))
            assigned_cluster[i] = np.argmin(euclid_dist)
        
        return assigned_cluster