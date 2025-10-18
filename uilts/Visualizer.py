# metrics_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.interpolate import make_interp_spline

class MetricsVisualizer:
    def __init__(self, save_dir='visuals'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_train_val(self, train_history, val_history, metrics=None):
        """
        绘制训练集和验证集共有指标曲线（平滑曲线，红色训练，蓝色验证）
        """
        if metrics is None:
            metrics = list(set(train_history.keys()) & set(val_history.keys()))
        for metric in metrics:
            train_values = np.array(train_history[metric])
            val_values = np.array(val_history[metric])
            epochs = np.arange(1, len(train_values) + 1)

            # 平滑曲线
            x_smooth = np.linspace(epochs.min(), epochs.max(), 200)
            train_smooth = make_interp_spline(epochs, train_values)(x_smooth)
            val_smooth = make_interp_spline(epochs, val_values)(x_smooth)

            plt.figure(figsize=(8, 5))
            plt.plot(x_smooth, train_smooth, color='red', label='Train', linewidth=2)
            plt.plot(x_smooth, val_smooth, color='blue', label='Validation', linewidth=2)
            plt.scatter(epochs, train_values, color='red', marker='o')  # 标出原始点
            plt.scatter(epochs, val_values, color='blue', marker='o')
            plt.title(f'Train vs Val {metric} over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'train_val_{metric}.png'))
            plt.close()

    def plot_test(self, test_metrics, filename='test_metrics.txt'):
        """
        绘制测试阶段指标，并保存文本总结
        test_metrics: dict, 可包含混淆矩阵及数值指标
        """
        # 混淆矩阵绘制
        cm = test_metrics.get('confusion_matrix', None)
        if cm is not None:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Test Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'test_confusion_matrix.png'))
            plt.close()

        # 整理文本输出
        lines = []
        for key, value in test_metrics.items():
            if key == 'confusion_matrix':
                continue
            elif key == 'classification_report':
                # classification_report 放最后
                continue
            elif isinstance(value, (int, float, np.number)):
                lines.append(f"{key}: {value:.4f}\n")
            else:
                lines.append(f"{key}: {value}\n")

        # classification_report 最后写
        if 'classification_report' in test_metrics:
            lines.append("\nClassification Report:\n")
            lines.append(test_metrics['classification_report'])
            lines.append("\n")

        # 保存文件
        file_path = os.path.join(self.save_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
