import torch
import torch.nn as nn
from torch.nn import functional as F


class PretrainLoss:
    """
    EEG 任务损失函数集合
    1. ECD_Loss: 情绪变化检测
    2. Channel_MLM_Loss: 通道掩码语言模型任务
    """
    def __init__(self, class_weights=None, label_smoothing=0.0):
        """
        :param class_weights: 分类任务类别权重
        :param label_smoothing: 分类任务标签平滑参数
        """
        super().__init__()
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device='cuda')
        self.CE = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def Channel_MLM_Loss(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Channel_MLM_Loss
        
        :param pred: 模型预测结果，shape (B, seq_len, n_cluster)
        :type pred: torch.Tensor
        :param target: kmeans 聚类后的软标签，shape (B, seq_len,)
        :type target: torch.Tensor
        """
        return F.cross_entropy(pred.view(-1, pred.shape[2]), target.view(-1).long())
    
    def ECD_Loss(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        ECD_Loss

        :param logits: 模型预测结果，shape (B, 2)
        :type logits: torch.Tensor
        :param targets: 情绪变化标签，shape (B,), labels (0: change, 1: no-change)
        :type targets: torch.Tensor
        :return: 计算得到的损失值
        :rtype: torch.Tensor
        """
        loss = self.CE(logits, targets.long())
        return loss