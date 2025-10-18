import torch
import torch.nn as nn
import numpy as np

class torch_kmeans(nn.Module):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, verbose=0, random_state=None, device=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _init_centroids(self, X):
        # KMeans++ 初始化
        n_samples, n_features = X.shape
        centroids = torch.empty((self.n_clusters, n_features), device=self.device)
        # 随机选择第一个中心
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        idx = torch.randint(0, n_samples, (1,))
        centroids[0] = X[idx]
        closest_dist_sq = torch.cdist(X, centroids[0:1]).squeeze() ** 2
        for c in range(1, self.n_clusters):
            probs = closest_dist_sq / closest_dist_sq.sum()
            idx = torch.multinomial(probs, 1)
            centroids[c] = X[idx]
            dist_sq = torch.cdist(X, centroids[c:c+1]).squeeze() ** 2
            closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)
        return centroids

    def fit(self, X):
        """
        X: torch.Tensor [n_samples, n_features]
        """
        X = X.to(self.device)
        n_samples = X.shape[0]
        centroids = self._init_centroids(X)
        for i in range(self.max_iter):
            # 计算距离并分配标签
            dists = torch.cdist(X, centroids)
            labels = torch.argmin(dists, dim=1)
            inertia = torch.sum(torch.min(dists, dim=1)[0] ** 2).item()
            # 更新中心
            new_centroids = torch.stack([
                X[labels == k].mean(dim=0) if (labels == k).sum() > 0 else centroids[k]
                for k in range(self.n_clusters)
            ])
            shift = torch.norm(centroids - new_centroids, dim=1).max().item()
            centroids = new_centroids
            if self.verbose:
                print(f"Iter {i+1}, inertia={inertia:.4f}, shift={shift:.6f}")
            if shift < self.tol:
                break
        self.cluster_centers_ = centroids.detach().cpu()
        self.labels_ = labels.detach().cpu()
        self.inertia_ = inertia
        self.n_iter_ = i + 1
        return self

    def predict(self, X):
        """
        X: torch.Tensor [n_samples, n_features]
        return: torch.LongTensor [n_samples]
        """
        X = X.to(self.device)
        centroids = self.cluster_centers_.to(self.device)
        dists = torch.cdist(X, centroids)
        return torch.argmin(dists, dim=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def transform(self, X):
        """
        返回每个样本到各中心的距离
        """
        X = X.to(self.device)
        centroids = self.cluster_centers_.to(self.device)
        return torch.cdist(X, centroids)

    def inertia(self, X=None):
        """
        计算总惯性（所有点到最近中心的距离平方和）
        """
        if X is None:
            return self.inertia_
        X = X.to(self.device)
        centroids = self.cluster_centers_.to(self.device)
        dists = torch.cdist(X, centroids)
        return torch.sum(torch.min(dists, dim=1)[0] ** 2).item()

    @property
    def n_clusters_(self):
        return self.n_clusters

    @property
    def cluster_centers(self):
        return self.cluster_centers_

    @property
    def labels(self):
        return self.labels_

    @property
    def inertia_(self):
        return self.inertia_

    @property
    def n_iter(self):
        return self.n_iter_
