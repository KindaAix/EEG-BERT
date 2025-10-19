from models.kmeans import torch_kmeans
import json
import torch
import numpy as np

def normalize_data(data):
    data_std = (data - data.mean(dim=0)) / (data.std(dim=0) + 1e-8)
    return data_std

class MultiNPYBatchedKMeans(torch_kmeans):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, verbose=0, 
                 random_state=None, device=None, batch_size=50000, 
                 files_batch_size=10, normalize_method='standard'):
        super().__init__(n_clusters, max_iter, tol, verbose, random_state, device)
        self.batch_size = batch_size
        self.files_batch_size = files_batch_size
        self.normalize_method = normalize_method

    def fit_from_npy_list(self, npy_paths, data_key='data'):
        """
        使用全部数据进行聚类，而不仅仅是采样数据
        """
        if self.verbose:
            print(f"开始处理 {len(npy_paths)} 个NPY文件")
            print(f"将使用全部数据进行聚类拟合")
        
        # 第一步：采样数据仅用于初始化中心点
        centroids = self._init_centroids_from_files(npy_paths, data_key)
        
        # 第二步：使用全部数据迭代优化
        for iter_num in range(self.max_iter):
            total_inertia = 0.0
            all_file_labels = []  # 存储每个文件的标签列表
            total_samples_processed = 0
            
            # 清空累积变量
            cluster_sums = [torch.zeros(centroids.shape[1], device=self.device) 
                           for _ in range(self.n_clusters)]
            cluster_counts = torch.zeros(self.n_clusters, device=self.device)
            
            # 处理所有文件的所有数据
            for file_batch_idx in range(0, len(npy_paths), self.files_batch_size):
                file_batch = npy_paths[file_batch_idx:file_batch_idx + self.files_batch_size]
                
                if self.verbose:
                    print(f"迭代 {iter_num+1}, 处理文件批次 {file_batch_idx//self.files_batch_size + 1}/{(len(npy_paths) + self.files_batch_size - 1) // self.files_batch_size}")
                
                # 处理当前文件批次，累积中心点更新信息
                batch_cluster_sums, batch_cluster_counts, batch_inertia, file_batch_labels = self._process_file_batch_for_fit(
                    file_batch, data_key, centroids)
                
                # 累积统计量
                for k in range(self.n_clusters):
                    cluster_sums[k] += batch_cluster_sums[k]
                    cluster_counts[k] += batch_cluster_counts[k]
                
                total_inertia += batch_inertia
                all_file_labels.extend(file_batch_labels)
                total_samples_processed += sum(len(labels) for labels in file_batch_labels)
            
            if self.verbose:
                print(f"迭代 {iter_num+1}, 已处理 {total_samples_processed} 个样本")
            
            # 使用全部数据的统计量更新中心点
            new_centroids = []
            for k in range(self.n_clusters):
                if cluster_counts[k] > 0:
                    new_centroids.append(cluster_sums[k] / cluster_counts[k])
                else:
                    new_centroids.append(centroids[k])
            
            new_centroids = torch.stack(new_centroids)
            
            # 检查收敛
            shift = torch.norm(centroids - new_centroids, dim=1).max().item()
            centroids = new_centroids
            
            if self.verbose:
                print(f"迭代 {iter_num+1}, 惯性={total_inertia:.4f}, 中心点偏移={shift:.6f}, 总样本数={total_samples_processed}")
            
            if shift < self.tol:
                if self.verbose:
                    print(f"在第 {iter_num+1} 次迭代收敛")
                break
        
        # 保存结果
        self.cluster_centers_ = centroids.detach().cpu()
        self.labels_ = all_file_labels  # 所有文件的标签
        self.inertia_ = total_inertia
        self.n_iter_ = iter_num + 1
        
        return self

    def _process_file_batch_for_fit(self, file_batch, data_key, centroids):
        """
        处理文件批次，返回累积统计量和标签
        """
        batch_cluster_sums = [torch.zeros(centroids.shape[1], device=self.device) 
                             for _ in range(self.n_clusters)]
        batch_cluster_counts = torch.zeros(self.n_clusters, device=self.device)
        total_inertia = 0.0
        file_batch_labels = []
        
        for npy_path in file_batch:
            try:
                with np.load(npy_path) as file_data:
                    if len(file_data) == 0:
                        file_batch_labels.append([])
                        continue
                    
                    # 处理整个文件的数据
                    processed_data = self._process_single_file_data(file_data)
                    file_tensor = torch.tensor(processed_data, device=self.device, dtype=torch.float32)
                    
                    # 计算距离和标签
                    with torch.no_grad():
                        dists = torch.cdist(file_tensor, centroids)
                        file_labels = torch.argmin(dists, dim=1)
                        total_inertia += torch.sum(torch.min(dists, dim=1)[0] ** 2).item()
                    
                    # 累积当前文件的统计量
                    for k in range(self.n_clusters):
                        mask = file_labels == k
                        if mask.sum() > 0:
                            batch_cluster_sums[k] += file_tensor[mask].sum(dim=0)
                            batch_cluster_counts[k] += mask.sum()
                    
                    file_batch_labels.append(file_labels.cpu())
                    
                    del file_tensor, dists, file_labels
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except Exception as e:
                if self.verbose:
                    print(f"警告: 处理文件 {npy_path} 时出错: {e}")
                file_batch_labels.append([])
                continue
        
        return batch_cluster_sums, batch_cluster_counts, total_inertia, file_batch_labels

    def _init_centroids_from_files(self, npy_paths, data_key, sample_per_file=1000):
        """仅用于初始化中心点，不用于拟合"""
        sampled_data = []
        
        for npy_path in npy_paths[:min(100, len(npy_paths))]:
            try:
                with np.load(npy_path) as file_data:
                        if len(file_data.shape) == 3 and all(dim > 0 for dim in file_data.shape):
                            processed_data = self._process_single_file_data(file_data)
    
                            # 采样用于初始化
                            n_samples = min(sample_per_file, len(processed_data))
                            indices = np.random.choice(len(processed_data), n_samples, replace=False)
                            sampled_data.append(processed_data[indices])
                            
            except Exception as e:
                if self.verbose:
                    print(f"警告: 初始化时无法读取文件 {npy_path}: {e}")
                continue
        
        if not sampled_data:
            raise ValueError("无法从任何文件中读取数据")
        
        all_sampled = np.concatenate(sampled_data, axis=0)
        init_data = torch.tensor(all_sampled, device=self.device, dtype=torch.float32)
        
        if self.verbose:
            print(f"初始化: 从 {len(sampled_data)} 个文件中采样 {len(all_sampled)} 个数据点用于初始化中心点")
            print(f"拟合: 将使用全部 {len(npy_paths)} 个文件的所有数据进行聚类")
        
        return self._init_centroids(init_data)


if __name__ == "__main__":
    with open('dataset/eeg_data_path_v1.2.json', 'r', encoding='utf-8') as f:
        eeg_data_path = json.load(f)
    
    kmeans = MultiNPYBatchedKMeans(
    n_clusters=8,
    batch_size=20000,
    files_batch_size=5,  # 每次处理5个文件
    max_iter=50,
    tol=1e-4,
    verbose=1,
    device='cuda'
)
    kmeans.fit_from_npy_list(eeg_data_path)

