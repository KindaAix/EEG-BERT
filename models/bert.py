import torch
from torch import nn
from encoder import BaseEncoder

class BERT(nn.Module):
    """
    bert
    """
    def __init__(self, config=None):
        super().__init__()
        self.encoder = BaseEncoder(
            seq_len=config.seq_len,
            dim=config.embedding_size,
            num_heads=config.num_heads,
            kernel_size=config.kernel_size,
            pool_len=config.pool_len,
            qkv_bias=config.qkv_bias,                                   
            attn_drop=config.attn_drop,                                   
            proj_drop=config.proj_drop,
            mlp_ratio=config.mlp_ratio,
            num_experts=config.num_experts,
            expert_capacity_factor=config.expert_capacity_factor,
            load_balancing_weight=config.load_balancing_weight
        )
        self.encoder_layers = nn.ModuleList(
            [self.encoder for _ in range(config.num_layers)]
        )
    def forward(self, x: torch.Tensor, aux_loss=0.0):
        for layer in self.encoder_layers:
            x, aux_loss = layer(x, aux_loss)
        return x, aux_loss
    

if __name__ == "__main__":
    x = torch.randn(64, 80, 1024)
    class Config:
        seq_len = 80
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERT(Config()).to(device)
    x = x.to(device)
    output = model(x)
    print(output[0].shape)
    print(f'{output[1].item():.4f}')