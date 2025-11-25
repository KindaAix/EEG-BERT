import torch
import torch.nn as nn
import torch.nn.functional as F
import swattention
from moe import MoEWithSharedExpert
from rope import RotaryPositionEmbedding

CUDA_NUM_THREADS = 64

class sw_1d_qkrpb_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, rpb, seq_len, kernel_size):
        attn_weight = swattention.qk_rpb_forward(query, key, rpb, seq_len, kernel_size, CUDA_NUM_THREADS)

        ctx.save_for_backward(query, key)
        ctx.seq_len, ctx.kernel_size = seq_len, kernel_size

        return attn_weight

    @staticmethod
    def backward(ctx, d_attn_weight):
        query, key = ctx.saved_tensors
        seq_len, kernel_size = ctx.seq_len, ctx.kernel_size

        d_query, d_key, d_rpb = swattention.qk_rpb_backward(d_attn_weight.contiguous(), query, key, seq_len, kernel_size, CUDA_NUM_THREADS)

        return d_query, d_key, d_rpb, None, None


class sw_1d_av_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_weight, value, seq_len, kernel_size):
        output = swattention.av_forward(attn_weight, value, seq_len, kernel_size, CUDA_NUM_THREADS)

        ctx.save_for_backward(attn_weight, value)
        ctx.seq_len, ctx.kernel_size = seq_len, kernel_size

        return output

    @staticmethod
    def backward(ctx, d_output):
        attn_weight, value = ctx.saved_tensors
        seq_len, kernel_size = ctx.seq_len, ctx.kernel_size

        d_attn_weight, d_value = swattention.av_backward(d_output.contiguous(), attn_weight, value, seq_len, kernel_size, CUDA_NUM_THREADS)

        return d_attn_weight, d_value, None, None


class BaseEncoder(nn.Module):
    def __init__(self, seq_len, dim=1024, num_heads=8, kernel_size=9, pool_len=8, qkv_bias=True, attn_drop=0.5, proj_drop=0.3, 
                 mlp_ratio=4.0, num_experts=4, expert_capacity_factor=1.0, load_balancing_weight=0.01):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert kernel_size % 2 == 1, "kernel size must be odd"
        self.kernel_size = kernel_size  # 9
        self.pool_len = pool_len  # 8
        self.local_len = kernel_size - 1  # 8

        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # Attention projection
        self.attn_proj = nn.Linear(dim, dim)
        self.attn_proj_drop = nn.Dropout(proj_drop)

        # MoE Feed-forward network
        self.moe = MoEWithSharedExpert(
            input_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            output_dim=dim,
            num_experts=num_experts,
            expert_capacity_factor=expert_capacity_factor,
            dropout=proj_drop,
            load_balancing_weight=load_balancing_weight
        )

        # Layer normalization (RMS-style normalization)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        # Components to generate pooled features
        self.pool = nn.AdaptiveAvgPool1d(self.pool_len)
        self.sr = nn.Conv1d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # Rotary Position Embedding
        self.rope = RotaryPositionEmbedding(dim=self.head_dim, max_seq_len=seq_len)

        # sequence length scale (simplified for 1D, using log scale)
        import numpy as np
        seq_length_scale = self.local_len + self.pool_len
        self.register_buffer("seq_length_scale", torch.tensor(np.log(seq_length_scale)), persistent=False)

        # Relative position bias for local attention
        self.relative_pos_bias_local = nn.Parameter(torch.zeros(num_heads, kernel_size))

    def forward(self, x, aux_loss=0.0):
        B, N, D = x.shape

        # Pre-LayerNorm for attention
        x_norm = self.norm1(x)

        with torch.amp.autocast("cuda", enabled=False):
            q = self.q(x_norm)  # (B, N, D)
            q_norm = F.normalize(q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
            del q
            
            q_norm_scaled = q_norm * F.softplus(self.temperature) * self.seq_length_scale

            kv = self.kv(x_norm)  # (B, N, 2*D)
            k_local = kv[..., :D].reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            v_local = kv[..., D:].reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            del kv 

            # Apply RoPE to query and key
            q_norm_scaled = self.rope(q_norm_scaled)
            k_local = self.rope(k_local)

            attn_local = sw_1d_qkrpb_cuda.apply(
                q_norm_scaled.contiguous(), 
                F.normalize(k_local, dim=-1).contiguous(),
                torch.zeros_like(self.relative_pos_bias_local), N, self.kernel_size
            )

            x_ = x_norm.transpose(1, 2).contiguous()  # (B, D, N)
            x_ = self.sr(x_)  # (B, D, N)
            x_ = self.act(x_)  # (B, D, N)
            x_ = self.pool(x_)  # (B, D, pool_len)
            x_ = x_.transpose(1, 2).contiguous()  # (B, pool_len, D)
            x_ = self.norm(x_)

            kv_pool = self.kv(x_)  # (B, pool_len, 2*D)
            k_pool = kv_pool[..., :D].reshape(B, self.pool_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            v_pool = kv_pool[..., D:].reshape(B, self.pool_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            del kv_pool, x_ 

            # Apply RoPE to pooled key
            k_pool = self.rope(k_pool)

            attn_pool = torch.matmul(q_norm_scaled, k_pool.transpose(-2, -1))  # 使用matmul而不是einsum

            attn = torch.cat([attn_local, attn_pool], dim=-1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            attn_local, attn_pool = torch.split(attn, [self.kernel_size, self.pool_len], dim=-1)

            x_local = sw_1d_av_cuda.apply(attn_local.contiguous(), v_local.contiguous(), N, self.kernel_size)

            x_pool = torch.matmul(attn_pool, v_pool)  # (B, num_heads, N, head_dim)
            x_pool = x_pool.transpose(1, 2).reshape(B, N, D)

            del attn, attn_local, attn_pool, q_norm_scaled, k_local, v_local, k_pool, v_pool

            x_local = x_local.transpose(1, 2).reshape(B, N, D)
            attn_output = x_local + x_pool

        # Attention projection and residual connection
        attn_output = self.attn_proj(attn_output)
        attn_output = self.attn_proj_drop(attn_output)
        x = x + attn_output

        # Pre-LayerNorm for MoE
        x_norm = self.norm2(x)
        moe_output, moe_metrics = self.moe(x_norm)
        x = x + moe_output

        # Accumulate auxiliary loss
        aux_loss += moe_metrics["aux_loss"]

        return x, aux_loss


if __name__ == "__main__":
    import torch

    batch_size = 32
    seq_len = 80
    dim = 1024
    num_heads = 8
    kernel_size = 9

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, dim, device=device)

    # Test single BaseEncoder block
    print("Testing BaseEncoder...")
    base_encoder = BaseEncoder(dim=dim, seq_len=seq_len, num_heads=num_heads, kernel_size=kernel_size).to(device)
    output_base, base_metrics = base_encoder(x)
    print("BaseEncoder output shape:", output_base.shape)
    print(f"BaseEncoder auxiliary loss: {base_metrics}")