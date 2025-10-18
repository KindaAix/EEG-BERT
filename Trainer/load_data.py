import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from models.kmeans import torch_kmeans
from glob import glob



class EEGClusterDataset(Dataset):
    def __init__(self, npz_dir, kmeans_path="kmeans.pt", kmeans_cls=torch_kmeans):
        """
        Args:
            npz_dir: 存放所有 npz 文件的目录
            kmeans_path: 预训练 KMeans 模型路径
        """
        npz_files = glob(f'{npz_dir}/**/*.npz', recursive=True)
        self.data =[]
        self.labels = []
        for path in npz_files:
            data = np.load(path ,allow_pickle=True)
            self.data.append(data['data'])
            self.labels.append(data['label'])
        # self.kmeans = self._load_kmeans(kmeans_path, kmeans_cls)

    def _load_kmeans(self, model_path="kmeans.pt", kmeans_cls=torch_kmeans):
        """加载预训练好的 KMeans 模型"""
        kmeans = kmeans_cls(n_clusters=256)  # 确认和训练时设置的一样
        state = torch.load(model_path, map_location="cpu")
        kmeans.load_state_dict(state)
        kmeans.eval()
        return kmeans

    def _seq_to_cluster_ids(self, data: np.ndarray):
        """
        将一条 EEG token 序列映射成 cluster_id 序列
        Args:
            data: numpy 数组, shape = (10, 310)  # 10帧，每帧310维特征
            kmeans: 已加载好的 torch_kmeans 模型
        Returns:
            cluster_ids: numpy 数组, shape = (10,)
        """
        assert len(data.shape) == 2, f"data 应该是 (10,310)，但得到 {data.shape}"
        
        # 转成 tensor
        x = torch.tensor(data, dtype=torch.float32)  # [10, 310]
        
        # 调用 KMeans 聚类
        with torch.no_grad():
            cluster_ids = self.kmeans.predict(x)  # [10]，每帧一个簇标签
        
        return cluster_ids.cpu().numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = int(self.labels[idx])
        
        # cluster_seq = self._seq_to_cluster_ids(data)  # shape (10,)
        
        # 返回 tensor 类型
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_dataloader(
    npz_dir: str,
    kmeans_path: str = "kmeans.pt",
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
):
    """
    通用 DataLoader 构建器
    Args:
        npz_dir: 存放 npz 文件的目录 (train/ valid/ test/)
        kmeans_path: 预训练 kmeans 模型路径
        batch_size: 批大小
        num_workers: DataLoader 线程数
        shuffle: 是否打乱数据
    """
    dataset = EEGClusterDataset(npz_dir=npz_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader

def get_train_loader(
    data_root="dataset/EEG-Tokens",
    kmeans_path="kmeans.pt",
    batch_size=64,
    num_workers=4,
):
    return get_dataloader(
        npz_dir=f"{data_root}/train",
        kmeans_path=kmeans_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,   # 训练集需要打乱
    )


def get_valid_loader(
    data_root="dataset/EEG-Tokens",
    kmeans_path="kmeans.pt",
    batch_size=64,
    num_workers=4,
):
    return get_dataloader(
        npz_dir=f"{data_root}/val",
        kmeans_path=kmeans_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # 验证集不打乱
    )


def get_test_loader(
    data_root="dataset/EEG-Tokens",
    kmeans_path="kmeans.pt",
    batch_size=64,
    num_workers=4,
):
    return get_dataloader(
        npz_dir=f"{data_root}/test",
        kmeans_path=kmeans_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # 测试集不打乱
    )