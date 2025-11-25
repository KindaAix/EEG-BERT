import torch
import torch.nn as nn
import math


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for 1D sequences
    Paper: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Create position indices
        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]

        # Create dimension indices (for even/odd dimensions)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))  # [dim//2]

        # Compute cos and sin for all positions and dimensions
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, dim] or [batch_size, num_heads, seq_len, head_dim]
        Returns:
            x with rotary position embedding applied
        """
        seq_len = x.size(-2)  # Get sequence length (second to last dimension)

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        # Get position embeddings for current sequence length
        pe = self.pe[:seq_len]  # [seq_len, dim]

        # Apply rotary embedding
        return self.apply_rotary_pos_emb(x, pe)

    def apply_rotary_pos_emb(self, x, pe):
        """
        Apply rotary position embedding to input tensor
        """
        if x.dim() == 3:  # [batch_size, seq_len, dim]
            # Split into even and odd dimensions
            x_even = x[..., 0::2]  # [batch_size, seq_len, dim//2]
            x_odd = x[..., 1::2]   # [batch_size, seq_len, dim//2]

            pe_even = pe[..., 0::2]  # [seq_len, dim//2]
            pe_odd = pe[..., 1::2]   # [seq_len, dim//2]

            # Apply rotation: x' = x * cos + rotate(x) * sin
            x_rotated = torch.cat([
                x_even * pe_even.cos() - x_odd * pe_odd.sin(),
                x_even * pe_even.sin() + x_odd * pe_odd.cos()
            ], dim=-1)

        elif x.dim() == 4:  # [batch_size, num_heads, seq_len, head_dim]
            # For multi-head attention case
            batch_size, num_heads, seq_len, head_dim = x.shape

            # Reshape to [batch_size * num_heads, seq_len, head_dim]
            x_reshaped = x.reshape(batch_size * num_heads, seq_len, head_dim)

            # Apply RoPE
            x_rotated = self.apply_rotary_pos_emb(x_reshaped, pe)

            # Reshape back
            x_rotated = x_rotated.view(batch_size, num_heads, seq_len, head_dim)

        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        return x_rotated


class SinusoidalPositionEmbedding(nn.Module):
    """
    Standard sinusoidal position embedding
    """

    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Create position embeddings
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            x + position embeddings
        """
        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        pe = self.pe[:seq_len]  # [seq_len, dim]
        return x + pe.unsqueeze(0)  # [1, seq_len, dim] + [batch_size, seq_len, dim]


class LearnedPositionEmbedding(nn.Module):
    """
    Learned position embedding
    """

    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pe = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            x + learned position embeddings
        """
        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        pe = self.pe(positions)  # [1, seq_len, dim]
        return x + pe


# Alias for backward compatibility
RoPE = RotaryPositionEmbedding
SinusoidalPE = SinusoidalPositionEmbedding
LearnedPE = LearnedPositionEmbedding


if __name__ == "__main__":
    # Test the position embeddings
    batch_size, seq_len, dim = 2, 10, 64

    # Test RoPE
    rope = RotaryPositionEmbedding(dim=dim, max_seq_len=512)
    x = torch.randn(batch_size, seq_len, dim)
    x_rope = rope(x)
    print(f"RoPE input shape: {x.shape}, output shape: {x_rope.shape}")

    # Test Sinusoidal
    sin_pe = SinusoidalPositionEmbedding(dim=dim, max_seq_len=512)
    x_sin = sin_pe(x)
    print(f"Sinusoidal PE input shape: {x.shape}, output shape: {x_sin.shape}")

    # Test Learned
    learned_pe = LearnedPositionEmbedding(dim=dim, max_seq_len=512)
    x_learned = learned_pe(x)
    print(f"Learned PE input shape: {x.shape}, output shape: {x_learned.shape}")

    print("Position embedding tests passed!")