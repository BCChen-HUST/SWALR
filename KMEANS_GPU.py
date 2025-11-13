from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

import torch
import numpy as np

class KMeansGPU:
    def __init__(self, num_clusters=8, init='k-means++', n_init=3, max_iter=100, tol=1e-4,
                 verbose=0, random_state=None, device='cuda', dtype=torch.float64):
        self.n_clusters = num_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.device = device
        self.dtype = dtype

    def _init_centroids(self, X, random_state):
        """Initialize centroids using k-means++ or random."""
        n_samples, n_features = X.shape

        if self.init == 'k-means++':
            centroids = torch.empty(self.n_clusters, n_features, device=self.device, dtype=self.dtype)

            first_idx = random_state.randint(0, n_samples)
            centroids[0] = X[first_idx]

            for i in range(1, self.n_clusters):
                dist_matrix = torch.cdist(X.to(self.dtype), centroids[:i].to(self.dtype))
                min_distances = torch.min(dist_matrix, dim=1)[0]
                distances_sq = min_distances ** 2

                probabilities = distances_sq / distances_sq.sum()
                cumulative_probs = probabilities.cumsum(dim=0)

                r = random_state.rand()
                idx = torch.searchsorted(cumulative_probs, r)
                idx = min(idx.item(), n_samples - 1)
                centroids[i] = X[idx]

        elif self.init == 'random':
            indices = torch.randperm(n_samples, device=self.device)[:self.n_clusters]
            centroids = X[indices]

        return centroids

    def _single_run(self, X, random_state):
        """single K-means run"""
        centroids = self._init_centroids(X, random_state)

        for i in range(self.max_iter):
            X = X.to(self.dtype)
            centroids = centroids.to(self.dtype)

            distances = torch.cdist(X, centroids)
            labels = torch.argmin(distances, dim=1)

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.n_clusters, device=self.device)

            new_centroids.scatter_reduce_(0, labels.unsqueeze(1).expand(-1, X.shape[1]), X,
                                          reduce='sum', include_self=False)
            counts.scatter_reduce_(0, labels, torch.ones_like(labels, dtype=torch.float),
                                   reduce='sum', include_self=False)

            empty_clusters = counts == 0
            if empty_clusters.any():
                empty_count = empty_clusters.sum().item()
                random_indices = torch.randperm(X.shape[0], device=self.device)[:empty_count]
                new_centroids[empty_clusters] = X[random_indices]
                counts[empty_clusters] = 1

            new_centroids = new_centroids / counts.unsqueeze(1)

            centroid_shift = torch.norm(new_centroids - centroids, dim=1).max()
            if centroid_shift < self.tol:
                break

            centroids = new_centroids

        distances = torch.cdist(X.to(self.dtype), centroids.to(self.dtype))
        min_distances = torch.gather(distances, 1, labels.unsqueeze(1)).squeeze()
        inertia = (min_distances ** 2).sum().item()

        return labels, centroids, inertia

    def fit(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        random_state = np.random.RandomState(self.random_state)

        best_inertia = float('inf')
        best_labels = None
        best_centroids = None

        for run in range(self.n_init):
            labels, centroids, inertia = self._single_run(X, random_state)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

                if inertia < self.tol * 10 and run >= 1:
                    if self.verbose:
                        print(f"Early stopping at run {run} with inertia {inertia:.4f}")
                    break

        self.labels_ = best_labels
        self.cluster_centers_ = best_centroids
        self.inertia_ = best_inertia

        return self.labels_, self.cluster_centers_

def plot_clusters(X, cluster_ids, centroids):
    if X.shape[1] < 2:
        print("dimension error")
        return

    plt.figure(figsize=(10, 8))

    # 使用更快的散点图绘制方法
    scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_ids, cmap='tab10',
                         s=10, alpha=0.6, marker='.')

    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x',
               s=200, linewidths=3, label='Centroids')

    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"K-Means Clustering (k={len(centroids)})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.savefig("/data1/BCChen/sw-ALR/picture/visual/kmeans.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # example
    X = np.random.rand(1000, 2)

    for i in range(4,10):
        num_clusters = i
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        kmeans_gpu = KMeansGPU(num_clusters= num_clusters, max_iter=100, random_state=2025, device=device)  # 'cuda' 表示使用 GPU
        labels, cluster_centers = kmeans_gpu.fit(X)
        labels= labels.cpu().numpy()
        cluster_centers= cluster_centers.cpu().numpy()

        print("Cluster centers:\n", cluster_centers)
        print("Labels:\n", labels)

        plot_clusters(X, labels, cluster_centers)

