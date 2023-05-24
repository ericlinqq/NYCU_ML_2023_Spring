import argparse
from dataloader import load_data
from kmeans import Kmeans
from kernel import user_defined_kernel
from utils import save_gif


def  parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./image1.png', type=str)
    parser.add_argument('--gif_path', default='./gif/image1', type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    img = load_data(args.data_path)
    print("kernel k means")
    kernel_k_means = Kmeans(kernel_function=user_defined_kernel, gamma_s=0.001, gamma_c=0.00007)
    for n_cluster in range(2, 5):
        print(f"k = {n_cluster}" + "-" * 40)
        cluster_result, visualization = kernel_k_means.fit(img, n_cluster=n_cluster, centroid_method='kmeans++', EPS=1e-9)
        save_gif(visualization, gif_path=args.gif_path, gif_name=f"kernel_kmeans_{n_cluster}.gif", duration=2)