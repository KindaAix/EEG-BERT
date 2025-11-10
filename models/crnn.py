import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG_CRNN(nn.Module):
    def __init__(self, n_clusters=256, embed_dim=128, n_classes=7, hidden_size=64, num_layers=1, dropout=0.5):
        """
        Args:
            n_clusters: 聚类簇数，用于 embedding 层
            embed_dim: embedding 维度
            n_classes: 分类数量
            hidden_size: RNN 隐藏层大小
            num_layers: RNN 层数
            dropout: dropout 比例
        """
        super(EEG_CRNN, self).__init__()
        # self.embedding = nn.Embedding(n_clusters, embed_dim)

        # CNN 提取空间特征
        # 假设输入 shape: (batch, channels=1, seq_len, embed_dim)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 310), padding=(1,0))  # 沿时间卷积
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(64)

        # RNN 捕捉时间依赖
        self.rnn = nn.GRU(input_size=64, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, n_classes)  # bidirectional *2

    def forward(self, x):
        """
        x: (batch, seq_len) 或 (batch, channels, seq_len) 取决于你的输入
        """
        # 如果输入是 cluster_id 序列 (batch, seq_len)
        if len(x.shape) == 3:
            # x = self.embedding(x)  # (batch, seq_len, embed_dim)
            x = x.unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        # elif len(x.shape) == 3:
        #     # (batch, channels, seq_len)
        #     batch, channels, seq_len = x.shape
        #     x = self.embedding(x)  # (batch, channels, seq_len, embed_dim)
        #     x = x.view(batch, 1, channels*seq_len, -1)  # (batch, 1, seq_len*channels, embed_dim)

        # CNN
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 32, seq_len, 1)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 64, seq_len, 1)
        x = x.squeeze(3)  # (batch, 64, seq_len)

        # 转置为 (batch, seq_len, feature) 送入 RNN
        x = x.permute(0, 2, 1)  # (batch, seq_len, 64)

        # RNN
        out, _ = self.rnn(x)  # (batch, seq_len, hidden*2)
        out = out[:, -1, :]   # 取最后一个时间步输出

        out = self.dropout(out)
        out = self.fc(out)  # (batch, n_classes)
        return out
    

if __name__ == "__main__":
    import torch
    from loss import EEGLoss
    from metrics import Metrics

    # ----------------- 设置 -----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEG_CRNN().to(device)
    print(model)

    # ----------------- Dummy 输入 -----------------
    batch_size = 8
    seq_len = 10
    n_classes = 7
    dummy_input = torch.randint(0, 256, (batch_size, seq_len)).to(device)  # cluster_id 序列
    dummy_labels = torch.randint(0, n_classes, (batch_size,)).to(device)

    # ----------------- 前向 -----------------
    output = model(dummy_input)  # (batch, n_classes)
    print("Output shape:", output.shape)

    # ----------------- 损失 -----------------
    criterion = EEGLoss()
    loss = criterion(output, dummy_labels)
    print("Loss:", loss.item())

    # ----------------- 初始化 Metrics -----------------
    metrics = Metrics(n_classes=n_classes)

    # 训练指标
    train_res = metrics.train_metrics(output, dummy_labels)
    print("\nTrain Metrics:")
    print(train_res)

    # 验证指标
    val_res = metrics.val_metrics(output, dummy_labels)
    print("\nValidation Metrics:")
    print(val_res)

    # 测试指标
    test_res = metrics.test_metrics(output, dummy_labels)
    print("\nTest Metrics:")
    print("Accuracy:", test_res['accuracy'])
    print("Top-2 Accuracy:", test_res['top_2_accuracy'])
    print("Precision:", test_res['precision'])
    print("Recall:", test_res['recall'])
    print("F1:", test_res['f1'])
    print("Cohen's Kappa:", test_res['cohen_kappa'])
    print("MCC:", test_res['mcc'])
    print("Confusion Matrix:\n", test_res['confusion_matrix'])
    print("Classification Report:\n", test_res['classification_report'])