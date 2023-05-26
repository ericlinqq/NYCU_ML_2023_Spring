from kmeans import Kmeans
from kernel import user_defined_kernel
import numpy as np
import matplotlib.pyplot as plt
import os

class SpectralClustering(Kmeans):
    def __init__(self, *, mode='unnormalized'):
        super().__init__()
        self.mode = mode
        self.U = None


    def _eigen_decomposition(self, W):
        D = np.diag(np.sum(W, axis=1))
        L = D - W

        print("eigen decomposition")
        if self.mode == 'unnormalized':
            eigenvalue, eigenvector = np.linalg.eig(L)
        elif self.mode == 'normalized':
            D_inverse_sqrt = np.diag(1 / np.diag(np.sqrt(D)))
            L_sym = D_inverse_sqrt @ L @ D_inverse_sqrt
            eigenvalue, eigenvector = np.linalg.eig(L_sym)

        sort_index = np.argsort(eigenvalue)
        self.U = eigenvector[:, sort_index]


    def _spectral_clustering(self, W, visualize=True):
        if self.U is None:
            self._eigen_decomposition(W)

        U = self.U[:, 1:self.n_cluster+1]

        if self.mode == 'normalized':
            sums = np.linalg.norm(U, axis=1).reshape(-1, 1)
            U = U / sums

        print("start k means")
        return self._kmeans(U, visualize=visualize)


    def fit(self, X, n_cluster=2, centroid_method='kmeans++', EPS=1e-9, visualize=True):
        if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
            raise Exception ('Input should be a similiarity matrix!')
        self.n_cluster = n_cluster
        self.centroid_method = centroid_method
        self.EPS = EPS

        return self._spectral_clustering(X, visualize=visualize)


    def plotEigenspaceLaplacian3D(self, assigned_cluster, plot_path='./plot', plot_name='eigenspace3D.png'):
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        ax = plt.figure().add_subplot(projection='3d')

        colors = ['r', 'g', 'b']
        x = self.U[:, 0]
        y = self.U[:, 1]
        z = self.U[:, 2]

        for c, i in zip(colors, np.arange(3)):
            ax.scatter(x[assigned_cluster==i], y[assigned_cluster==i], z[assigned_cluster==i], c=c)

        ax.set_xlabel('1st eigenvector')
        ax.set_ylabel('2nd eigenvector')
        ax.set_zlabel('3rd eigenvector')

        plt.savefig(os.path.join(plot_path, plot_name))


    def plotEigenspaceLaplacian2D(self, assigned_cluster, plot_path='./figure', plot_name='eigenspace2D.png'):
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        fig = plt.figure()

        colors = ['r', 'g']
        x = self.U[:, 0]
        y = self.U[:, 1]

        for c, i in zip(colors, np.arange(2)):
            plt.scatter(x[assigned_cluster==i], y[assigned_cluster==i], c=c)

        plt.xlabel('1st eigenvector')
        plt.ylabel('2nd eigenvector')

        plt.savefig(os.path.join(plot_path, plot_name))
