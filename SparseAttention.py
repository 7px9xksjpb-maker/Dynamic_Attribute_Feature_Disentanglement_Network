import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAttentionCompressor(nn.Module):
    """
    基于注意力的稀疏降维前端模块。
    通过学习每个维度的重要性，动态挑选出最核心的 M_comp 个物理特征点，并保持其原始空间顺序。
    """
    def __init__(
        self, 
        input_bins: int, 
        output_bins: int = 128, 
        hidden_channels: int = 32
    ):
        super().__init__()
        self.input_bins = input_bins
        self.output_bins = output_bins
        
        if output_bins >= input_bins:
            raise ValueError(f"output_bins ({output_bins}) 必须严格小于 input_bins ({input_bins})")

        # 1. 打分网络：提取局部上下文以评估该距离单元的重要性
        self.scorer = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(4, hidden_channels), num_channels=hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1)
        )
        
        # 可选：对最终输出进行特征规范化，稳定后续的 LSTM 和 GP 先验
        self.out_norm = nn.LayerNorm(output_bins)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, T, M_raw]
        return: [B, T, M_comp]
        """
        if X.dim() != 3:
            raise ValueError(f'Expected [B, T, M], got {tuple(X.shape)}.')
            
        B, T_, M_ = X.shape
        x_flat = X.reshape(B * T_, 1, M_)  # 合并 B 和 T 维度以复用 1D 卷积

        # --- 第一步：计算注意力得分 ---
        # logits: [B*T, 1, M_raw]
        logits = self.scorer(x_flat)
        logits = logits.squeeze(1) # [B*T, M_raw]
        
        # 归一化得分 (0~1)，代表保留该特征点的权重
        attn_weights = torch.sigmoid(logits)

        # --- 第二步：稀疏选择 (Top-K) ---
        # 选取得分最高的 output_bins 个位置
        # top_weights, top_indices shape: [B*T, output_bins]
        top_weights, top_indices = torch.topk(attn_weights, self.output_bins, dim=-1)

        # --- 第三步：空间位置对齐 (至关重要) ---
        # top_indices 是按分数降序排列的，我们需要将其重新按空间位置（即索引值大小）升序排列
        # sort_args 是升序排列后的元素在原 top_indices 中的位置
        sorted_indices, sort_args = torch.sort(top_indices, dim=-1)
        
        # 对权重也进行相应的重新排序，使其与对齐后的特征一一对应
        sorted_weights = torch.gather(top_weights, dim=-1, index=sort_args)

        # --- 第四步：特征聚合与可微分调制 ---
        x_flat_squeeze = X.reshape(B * T_, M_)
        
        # 物理意义：从原始雷达回波中，无损地“抠”出这 output_bins 个点
        gathered_features = torch.gather(x_flat_squeeze, dim=-1, index=sorted_indices)
        
        # 数学意义：乘以得分，使得梯度能够流回打分网络 self.scorer
        compressed_x = gathered_features * sorted_weights
        
        # 恢复 [B, T, M_comp] 形状
        compressed_x = compressed_x.view(B, T_, self.output_bins)
        
        # 归一化输出（有利于与原版的均值池化在数值分布上保持一致）
        compressed_x = self.out_norm(compressed_x)

        return compressed_x