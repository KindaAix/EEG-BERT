import torch
import random


def random_mask(x: torch.Tensor, mask_prob=0.08, mask_length=10, noise=None):
    """
    随机掩码函数，采用 HuBERT 的掩码策略，对每个通道做同样的操作。
    
    x: (B, 160, 744) -> 744 = 12 channels * 62 features
    Mask strategy: HuBERT-style masking
    - 随机选择起始点，掩码长度固定为 mask_length (默认 10)
    - 每个时间步有 mask_prob 概率成为掩码起始点
    - 对所有12个通道应用相同的掩码模式
    被掩码的部分将传递指定的噪声信号。
    
    :param x: 输入张量，形状 (B, 160, 744)
    :param mask_prob: 掩码起始点概率，默认 0.08
    :param mask_length: 掩码长度，默认 10
    :param noise: 指定的噪声信号，如果 None，则使用高斯噪声
    :return: x_masked
    """
    x = x.reshape(x.shape[0], x.shape[1], 12, -1).transpose(2, 3)  # 保持原始形状 (B, 160, 62, 12)
    B, seq_len, e_channels, band = x.shape  # (B, 160, 62, 12)
    
    if noise is None:
        noise = torch.randn(seq_len, e_channels, dtype=x.dtype, device=x.device)
    noise = noise.to(x.device)
    # HuBERT 掩码策略：每个时间步都可能是掩码起始点
    mask_indices = []
    for _ in range(62):
        mask_in = []
        i = 0
        while i < seq_len:
            if random.random() < mask_prob:
                end = min(i + mask_length, seq_len)
                mask_in.append((i, end))
                i = end
            else:
                i += 1
        mask_indices.append(mask_in)
    for i in range(62):
        for start, end in mask_indices[i]:
            x[:, start:end, i, :] = noise[:end-start, :].unsqueeze(0).expand(B, end-start, band)
    return x.transpose(2, 3).reshape(B, seq_len, -1)  # (B, 160, 744)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(128, 160, 744, device=device)  # (B, seq_len, n_features)
    noise = torch.randn(10, 12, device=device)  # (mask_length, e_channels)
    print(noise.shape)
    x_masked = random_mask(x, noise=noise)
    print("Original x shape:", x.shape)
    print("Masked x shape:", x_masked.shape)