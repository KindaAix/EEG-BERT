import random
import torch
from torch import nn
from bert import BERT


class BERTLM(nn.Module):
    def __init__(self, bert: BERT):
        super().__init__()
        self.bert = bert
        self.cmlm = ChannelMLMTask(hidden_dim=bert.config.embedding_size, out_dim=bert.config.n_cluster)
        self.ess = ESSTask(hidden_dim=bert.config.seq_len)

    def forward(self, x: torch.Tensor, aux_loss=0.0):
        x, aux_loss, feature = self.bert(x, aux_loss)
        cmlm_output = self.cmlm(x)
        ess_output = self.ess(x)
        return cmlm_output, ess_output, aux_loss, feature


class ChannelMLMTask(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.MLP = nn.Linear(hidden_dim, out_dim)
    
    def mask_features(self, x: torch.Tensor):
        """
        x: (B, seq_len, hidden_dim)
        Mask strategy: 6 features (delta/theta/alpha/beta/gamma/ce) independently mask
        Each mask 50% probability per trial, synchronized across large/small windows
        Mask length 4-8 continuous frames
        """
        B, seq_len, _ = x.shape
        x_masked = x.clone()
        mask_matrix = torch.zeros_like(x, dtype=torch.bool)
        for f_idx in range(6):  # 6 features
            if random.random() < self.mask_prob:
                # choose start index for consecutive mask
                start = random.randint(0, seq_len - self.mask_len_max)
                length = random.randint(self.mask_len_min, self.mask_len_max)
                idx_range = slice(start, start + length)
                # mask large and small windows (f_idx and f_idx+6)
                cols = [f_idx, f_idx + 6]
                for c in cols:
                    mask_matrix[:, idx_range, c] = True
                    x_masked[:, idx_range, c] = self.noise[:length]
        return x_masked, mask_matrix
    
    def forward(self, x: torch.Tensor):
        """
        x: (B, seq_len, hidden_dim)
        Returns:
            Channel-MLM output: (B, seq_len, out_dim)
        """
        ChannelMLMTaskOutput = self.MLP(x)
        return ChannelMLMTaskOutput


class ESSTask(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor):
        """
        cls_x: (B, hidden_dim)
        Returns logits: (B, 2)
        Masks first frame by design
        """
        x = self.pool(x).squeeze(-1)  # (B, hidden_dim)
        logits = self.mlp(x)
        return logits


if __name__ == "__main__":
    x = torch.randn(16, 160, 744)  # (B, seq_len, n_features)
    class Config:
        seq_len = 160
        embedding_size = 1024
        num_heads = 8
        kernel_size = 9
        pool_len = 8
        qkv_bias = True
        attn_drop = 0.5
        proj_drop = 0.3
        mlp_ratio = 4.0
        num_experts = 4
        expert_capacity_factor = 1.0
        load_balancing_weight = 0.01
        num_layers = 12
        n_cluster = 128
        n_features = 744

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = BERT(Config()).to(device)
    bert_lm = BERTLM(bert_model).to(device)
    output = bert_lm(x.to(device))
    print(f"\ncmlm output shape: {output[0].shape}, \ness output shape: {output[1].shape}, \naux_loss: {output[2]:.4f}")