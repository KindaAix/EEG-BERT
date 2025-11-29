import torch
import torch.nn as nn

class torch_kmeans(nn.Module):
    def __init__(self, n_clusters=128, max_iter=5, tol=1e-4, verbose=0, random_state=None, device=None):
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
        n_samples, n_features = X.shape
        centroids = torch.empty((self.n_clusters, n_features), device=self.device)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        else:
            torch.manual_seed(42)  # 默认种子
        idx = torch.randint(0, n_samples, (1,))
        centroids[0] = X[idx]
        closest_dist_sq = torch.cdist(X, centroids[0:1]).squeeze() ** 2
        for c in range(1, self.n_clusters):
            probs = closest_dist_sq / (closest_dist_sq.sum() + 1e-10)
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
            dists = torch.cdist(X, centroids)
            labels = torch.argmin(dists, dim=1)
            inertia = torch.sum(torch.min(dists, dim=1)[0] ** 2).item()
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(dim=0)
                else:
                    idx = torch.randint(0, n_samples, (1,))
                    new_centroids[k] = X[idx]
            shift = torch.norm(centroids - new_centroids, dim=1).max().item()
            centroids = new_centroids
            if self.verbose:
                print(f"Iter {i+1}, inertia={inertia:.4f}, shift={shift:.6f}")
            if shift < self.tol:
                break
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = i + 1
        return self

    def predict(self, X):
        """
        X: torch.Tensor [n_samples, n_features]
        return: torch.Tensor [n_samples]
        """
        if X.device != self.device:
            X = X.to(self.device)
        dists = torch.cdist(X, self.cluster_centers_)
        return torch.argmin(dists, dim=1) if self.device == "cuda" else torch.argmin(dists, dim=1).cpu()

    def fit_and_predict_labels(self, X):
        """
        先拟合数据（最多 max_iter 轮），然后返回输入数据的簇序号（硬标签）
        X: torch.Tensor [n_samples, n_features] 或 [batch_size, seq_len, n_features]
        return: torch.Tensor [n_samples] 或 [batch_size, seq_len]
        """
        if len(X.shape) == 3:
            bs, seq_len, _ = X.shape
            X = X.reshape(-1, X.shape[-1])  # [Batch_Size*Seq_len, n_features]
            is_3d = True
        else:
            is_3d = False
        self.fit(X)
        labels = self.labels_
        if is_3d:
            labels = labels.reshape(bs, seq_len)
        return labels if self.device == "cuda" else labels.cpu()

    def transform(self, X):
        """
        返回每个样本到各中心的距离
        """
        if X.device != self.device:
            X = X.to(self.device)
        centroids = self.cluster_centers_
        return torch.cdist(X, centroids) if self.device == "cuda" else torch.cdist(X, centroids).cpu()

    def inertia(self, X=None):
        """
        计算总惯性（所有点到最近中心的距离平方和）
        如果 X 为 None，返回训练时的惯性
        """
        if X is None:
            return self.inertia_
        X = X.to(self.device)
        centroids = self.cluster_centers_
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
