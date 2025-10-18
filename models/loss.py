import torch
import torch.nn as nn

class EEGLoss(nn.Module):
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
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        """
        Args:
            logits: (batch, n_classes) RCNN 输出
            labels: (batch,) long 类型标签，范围 0~n_classes-1
        Returns:
            loss: 标量
        """
        return self.criterion(logits, labels)
