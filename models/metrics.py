# metrics.py
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, classification_report
import numpy as np

class Metrics:
    def __init__(self, n_classes=7):
        self.n_classes = n_classes

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise TypeError("Input must be torch.Tensor or np.ndarray")

    # ----------------- 基础指标 -----------------
    def accuracy(self, y_true, y_pred):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred, average='macro'):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return precision_score(y_true, y_pred, average=average, zero_division=0)

    def recall(self, y_true, y_pred, average='macro'):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return recall_score(y_true, y_pred, average=average, zero_division=0)

    def f1(self, y_true, y_pred, average='macro'):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return f1_score(y_true, y_pred, average=average, zero_division=0)

    def confusion_matrix(self, y_true, y_pred):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return confusion_matrix(y_true, y_pred, labels=np.arange(self.n_classes))

    # ----------------- 扩展指标 -----------------
    def top_k_accuracy(self, logits, y_true, k=2):
        y_true = self._to_numpy(y_true)
        if isinstance(logits, torch.Tensor):
            topk_pred = torch.topk(logits, k=k, dim=1)[1]
            topk_pred = topk_pred.cpu().numpy()
        else:
            topk_pred = np.argsort(logits, axis=1)[:, -k:]
        correct = np.any(topk_pred == y_true[:, None], axis=1)
        return np.mean(correct)

    def cohen_kappa(self, y_true, y_pred):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return cohen_kappa_score(y_true, y_pred)

    def mcc(self, y_true, y_pred):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return matthews_corrcoef(y_true, y_pred)

    def classification_report(self, y_true, y_pred):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        return classification_report(y_true, y_pred, labels=np.arange(self.n_classes))

    # ----------------- 分场景调用 -----------------
    def train_metrics(self, logits, y_true):
        """训练时调用，只计算 loss/accuracy/Top-K"""
        y_pred = torch.argmax(logits, dim=1)
        return {
            'accuracy': self.accuracy(y_true, y_pred),
            'top_2_accuracy': self.top_k_accuracy(logits, y_true, k=2)
        }

    def val_metrics(self, logits, y_true):
        """验证时调用，计算常规指标 + Top-K"""
        y_pred = torch.argmax(logits, dim=1)
        return {
            'accuracy': self.accuracy(y_true, y_pred),
            'precision': self.precision(y_true, y_pred),
            'recall': self.recall(y_true, y_pred),
            'f1': self.f1(y_true, y_pred),
            'top_2_accuracy': self.top_k_accuracy(logits, y_true, k=2)
        }

    def test_metrics(self, logits, y_true):
        """测试/评估时调用，输出所有指标"""
        y_pred = torch.argmax(logits, dim=1)
        return {
            'accuracy': self.accuracy(y_true, y_pred),
            'precision': self.precision(y_true, y_pred),
            'recall': self.recall(y_true, y_pred),
            'f1': self.f1(y_true, y_pred),
            'confusion_matrix': self.confusion_matrix(y_true, y_pred),
            'cohen_kappa': self.cohen_kappa(y_true, y_pred),
            'mcc': self.mcc(y_true, y_pred),
            'classification_report': self.classification_report(y_true, y_pred),
            'top_2_accuracy': self.top_k_accuracy(logits, y_true, k=2)
        }
