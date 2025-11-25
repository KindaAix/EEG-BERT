import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Router(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, capacity_factor: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
        Returns:
            gate_weights: 门控权重 [batch_size, seq_len, num_experts]
            expert_indices: 专家索引 [batch_size, seq_len]
        """
        # 计算门控权重
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # 选择top-1专家
        expert_weights, expert_indices = torch.topk(gate_weights, k=1, dim=-1)
        expert_indices = expert_indices.squeeze(-1)  # [batch_size, seq_len]
        
        return gate_weights, expert_indices

class MoEWithSharedExpert(nn.Module):    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 4,
        expert_capacity_factor: float = 1.0,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.load_balancing_weight = load_balancing_weight
        
        self.router = Router(input_dim, num_experts, expert_capacity_factor)
        
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        self.shared_expert = Expert(input_dim, hidden_dim, output_dim, dropout)
        
        self.shared_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
        Returns:
            output: 输出张量 [batch_size, seq_len, hidden_dim]
            aux_loss: 辅助损失用于负载均衡
        """        
        gate_weights, expert_indices = self.router(x)
        
        final_output = torch.zeros_like(x)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx)  # [batch_size, seq_len]
            if expert_mask.any():
                expert_input = x[expert_mask]  # [num_tokens, hidden_dim]
                expert_output = self.experts[expert_idx](expert_input)
                final_output[expert_mask] += expert_output
        
        shared_output = self.shared_expert(x)  # [batch_size, seq_len, hidden_dim]
        
        router_confidence = gate_weights.max(dim=-1)[0]  # [batch_size, seq_len]
        shared_weights = (1 - router_confidence).unsqueeze(-1) * torch.sigmoid(self.shared_weight)
        
        final_output = final_output + shared_weights * shared_output
        
        # 计算负载均衡损失
        aux_loss = self._compute_auxiliary_loss(gate_weights, expert_indices)
        
        return final_output, {"aux_loss": aux_loss, "gate_weights": gate_weights}
    
    def _compute_auxiliary_loss(
        self, 
        gate_weights: torch.Tensor, 
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """计算负载均衡辅助损失"""
        batch_size, seq_len, num_experts = gate_weights.shape
        
        expert_mask = F.one_hot(expert_indices, num_classes=num_experts)  # [batch_size, seq_len, num_experts]
        expert_usage = expert_mask.float().mean(dim=(0, 1))  # [num_experts]
        
        gate_weights_mean = gate_weights.mean(dim=(0, 1))  # [num_experts]
        
        load_balancing_loss = torch.sum(expert_usage * gate_weights_mean) * num_experts
        
        return self.load_balancing_weight * load_balancing_loss
    
    def get_expert_usage(self, x: torch.Tensor) -> Dict:
        """获取专家使用统计"""
        with torch.no_grad():
            gate_weights, expert_indices = self.router(x)
            batch_size, seq_len = expert_indices.shape
            
            expert_usage = {}
            for expert_idx in range(self.num_experts):
                usage_count = (expert_indices == expert_idx).sum().item()
                usage_ratio = usage_count / (batch_size * seq_len)
                expert_usage[f"expert_{expert_idx}"] = {
                    "count": usage_count,
                    "ratio": usage_ratio
                }
            
            expert_usage["shared_expert"] = {
                "count": batch_size * seq_len,
                "ratio": 1.0
            }
            
            return expert_usage


if __name__ == "__main__":
    input_dim = 1024
    hidden_dim = 3048
    output_dim = 1024
    batch_size = 128
    seq_len = 80
    
    model = MoEWithSharedExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=4,
        load_balancing_weight=0.01
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output, metrics = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"辅助损失: {metrics['aux_loss'].item():.6f}")
    
    usage_stats = model.get_expert_usage(x)
    print("\n专家使用情况:")
    for expert, stats in usage_stats.items():
        print(f"  {expert}: {stats['ratio']:.3f} ({stats['count']} tokens)")
    
    gate_weights = metrics['gate_weights']
    print(f"\n门控权重统计:")
    print(f"  均值: {gate_weights.mean().item():.4f}")
    print(f"  标准差: {gate_weights.std().item():.4f}")
    print(f"  最大置信度: {gate_weights.max().item():.4f}")
    print(f"  最小置信度: {gate_weights.min().item():.4f}")