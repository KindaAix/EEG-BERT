import torch
from torch import nn

from attention_toolkit import SlidingWindowAttention, PoolingAttention

class BERT_arch(nn.Module):
    """
    bert - 未来改成滑动窗口注意力加池化注意力，调用attention.py中的模块
    """
    def __init__(self, input_dim: int = 744, embed_dim: int = 128, num_heads: int = 8, hidden_dim: int = 512, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_encoder = Encoder()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sliding_attention = SlidingWindowAttention(window_size=16, overlap=8)
        self.pooling_attention = PoolingAttention(pool_size=4)

    def forward(self, x: torch.Tensor):
        pass