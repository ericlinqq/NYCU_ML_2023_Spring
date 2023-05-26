import argparse
from dataloader import load_data
from kmeans import Kmeans
from spectral_clustering import SpectralClustering
from kernel import user_defined_kernel
from utils import save_gif


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./image1.png', type=str)
    parser.add_argument('--gif_path', default='./gif/image1', type=str)
    parser.add_argument('--plot_path', default='./plot/image1', type=str)
    parser.add_argument('--gamma_s', default=0.001, type=float)
    parser.add_argument('--gamma_c', default=0.001, type=float)
    parser.add_argument('--centroid_method', default='kmeans++', choices=['kmeans++', 'random'], type=str)
    parser.add_argument('--EPS', default=1e-9, type=float)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    img, height, width = load_data(args.data_path)
    W = user_defined_kernel(img, width=width, gamma_s=args.gamma_s, gamma_c=args.gamma_c)

    print("kernel k means")
    kernel_k_means = Kmeans()
    for n_cluster in range(2, 5):
        print("-" * 40 + f"k = {n_cluster}" + "-" * 40)
        cluster_result, assigned_color = kernel_k_means.fit(W, n_cluster=n_cluster, centroid_method=args.centroid_method, EPS=args.EPS, visualize=True)
        visualization = assigned_color.reshape(-1, height, width, assigned_color.shape[-1])
        save_gif(visualization, gif_path=args.gif_path, gif_name=f"kernel_kmeans_{args.centroid_method}_{n_cluster}.gif", duration=2)


    print("unnormalized spectral clustering")
    unnormalized_SC = SpectralClustering(mode='unnormalized')
    for n_cluster in range(2, 5):
        print("-" * 40 + f"k = {n_cluster}" + "-" * 40)
        cluster_result, assigned_color = unnormalized_SC.fit(W, n_cluster=n_cluster, centroid_method=args.centroid_method, EPS=args.EPS, visualize=True)
        visualization = assigned_color.reshape(-1, height, width, assigned_color.shape[-1])
        save_gif(visualization, gif_path=args.gif_path, gif_name=f"unnormalized_SC_{args.centroid_method}_{n_cluster}.gif", duration=2)

        if n_cluster == 2:
            unnormalized_SC.plotEigenspaceLaplacian2D(assigned_cluster=cluster_result, plot_path=args.plot_path, plot_name=f"unnormalized_SC_{args.centroid_method}_{n_cluster}.png")
        elif n_cluster == 3:
            unnormalized_SC.plotEigenspaceLaplacian3D(assigned_cluster=cluster_result, plot_path=args.plot_path, plot_name=f"unnormalized_SC_{args.centroid_method}_{n_cluster}.png")


    print("normalized spectral clustering")
    normalized_SC = SpectralClustering(mode='normalized')
    for n_cluster in range(2, 5):
        print("-" * 40 + f"k = {n_cluster}" + "-" * 40)
        cluster_result, assigned_color = normalized_SC.fit(W, n_cluster=n_cluster, centroid_method=args.centroid_method, EPS=args.EPS, visualize=True)
        visualization = assigned_color.reshape(-1, height, width, assigned_color.shape[-1])
        save_gif(visualization, gif_path=args.gif_path, gif_name=f"normalized_SC_{args.centroid_method}_{n_cluster}.gif", duration=2)

        if n_cluster == 2:
            normalized_SC.plotEigenspaceLaplacian2D(assigned_cluster=cluster_result, plot_path=args.plot_path, plot_name=f"normalized_SC_{args.centroid_method}_{n_cluster}.png")
        elif n_cluster == 3:
            normalized_SC.plotEigenspaceLaplacian3D(assigned_cluster=cluster_result, plot_path=args.plot_path, plot_name=f"normalized_SC_{args.centroid_method}_{n_cluster}.png")
