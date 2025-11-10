import random
import torch
from torch import nn
import numpy as np
from models.bert_arch import BERT_arch as bert
from models.embedding import BERTEmbedding


class ChannelMLMTask(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, mask_prob: float = 0.5, mask_len_min: int = 4, mask_len_max: int = 8, seed: int = 42):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.mask_prob = mask_prob
        self.mask_len_min = mask_len_min
        self.mask_len_max = mask_len_max
        random.seed(seed)
        torch.manual_seed(seed)
        self.noise = torch.tensor(np.random.normal(0, 1, size=(mask_len_max,)), dtype=torch.float32)
        self.MLP = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim // 2, self.mlp_dim)
        )
    
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
            Channel-MLM output: (B, seq_len, hidden_dim)
        """
        ChannelMLMTaskOutput = self.MLP(x)
        return ChannelMLMTaskOutput


class ESSTask(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, cls_x: torch.Tensor):
        """
        cls_x: (B, hidden_dim)
        Returns logits: (B, 2)
        Masks first frame by design
        """
        logits = self.mlp(cls_x)
        return logits


class PretrainTask(nn.Module):
    def __init__(self, hidden_dim: int,):
        super(PretrainTask).__init__()
        self.embedding = BERTEmbedding()
        self.encoder = bert()
        self.CMLM = ChannelMLMTask()
        self.ESS = ESSTask()

        
    def forward(self, x: torch.Tensor):
        x_masked, _ = self.CMLM.mask_features(x)
        x_encoder_output = self.encoder(x_masked)
        cmlm_out = self.CMLM(x_encoder_output[:, 1:,:])
        ess_out = self.ESS(x_encoder_output[:, 0,:].squeeze(1))
        return cmlm_out, ess_out