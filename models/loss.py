import torch
import torch.nn as nn
from torch.nn import functional as F


class EEGLoss:
    """
    EEG 任务损失函数集合
    1. CrossEntropyLoss: 分类任务
    2. Channel_MLM_Loss: 通道掩码语言模型任务
    3. ESS_Loss: 状态变化检测任务
    """
    def __init__(self, n_classes=7, class_weights=None, label_smoothing=0.0):
        """
        Args:
            n_classes: 分类数量
            class_weights: 可选，每类权重 tensor 或 list，形状 (n_classes,)
            label_smoothing: 标签平滑系数，0 表示不使用
        """
        super(EEGLoss, self).__init__()
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device='cuda')
        self.CE = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.MSE = nn.MSELoss()
        self.BCE = nn.BCEWithLogitsLoss()

    def CrossEntropyLoss(self, logits, labels):
        """
        Args:
            logits: (batch, n_classes) RCNN 输出
            labels: (batch,) long 类型标签，范围 0~n_classes-1
        Returns:
            loss: 标量
        """
        return self.CE(logits, labels)
    
    def Channel_MLM_Loss(self, pred: torch.Tensor, target: torch.Tensor):
        # 这里需要改成kmeans++的损失
        return self.MSE(pred, target)
    
    def ESS_Loss(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: (B, seq_len, 2)
        targets: (B, seq_len) 0/1 labels (0: change, 1: no-change)
        """
        # 这里需要该成分类损失
        B, seq_len = targets.shape
        targets_onehot = F.one_hot(targets.long(), num_classes=2).float()
        loss = self.BCE(logits.view(-1,2), targets_onehot.view(-1,2))
        return loss

    def loss(self, x, cmlm_target, ess_target):
        return 0.6 * self.Channel_MLM_Loss(x[:, 1:, :], cmlm_target) + 0.4 * self.ESS_Loss(x[:, 0, :].squeeze(1), ess_target)