import torch
import torch.nn as nn
import torch.nn.functional as F


def align_tokens(x, target_len):
    """
    将 token 数调整为 target_len，维持 batch 和 channel 不变
    输入: x [B, N, C]，输出: [B, target_len, C]
    """
    B, N, C = x.shape
    x = x.permute(0, 2, 1)  # [B, C, N]
    x = F.adaptive_avg_pool1d(x, target_len)  # [B, C, target_len]
    x = x.permute(0, 2, 1)  # [B, target_len, C]
    return x

# -----------------------------
# 1. MoE 融合模块（融合 Mamba + RAD-DINO）
# -----------------------------


class MoEFusion(nn.Module):
    """
    Mixture-of-Experts Fusion Module for fusing Mamba and RAD-DINO features.
    Supports dynamic gating to weight each expert per token.
    """
    def __init__(self, mamba_dim=1024, rad_dino_dim=768, fused_dim=1024, hidden_dim=None):
        super().__init__()

        
        self.mamba_proj = nn.Identity() if mamba_dim == fused_dim else nn.Linear(mamba_dim, fused_dim)
        self.raddino_proj = nn.Identity() if rad_dino_dim == fused_dim else nn.Linear(rad_dino_dim, fused_dim)
        
        
        gate_input_dim = fused_dim * 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim or gate_input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim or gate_input_dim, 2),  # 2 experts
            nn.Softmax(dim=-1)
        )

    def forward(self, mamba_feat, raddino_feat):
        """
        Args:
            mamba_feat: [B, N, mamba_dim]
            raddino_feat: [B, N, rad_dino_dim]

        Returns:
            fused_feat: [B, N, fused_dim]
        """
        # Step 1: 投影到统一维度
        m_proj = self.mamba_proj(mamba_feat)       # [B, N, fused_dim]
        r_proj = self.raddino_proj(raddino_feat)   # [B, N, fused_dim]

        if m_proj.shape[1] != r_proj.shape[1]:
            target_len = min(m_proj.shape[1], r_proj.shape[1])
            m_proj = align_tokens(m_proj, target_len)
            r_proj = align_tokens(r_proj, target_len)

        # Step 2: 拼接后输入门控网络
        concat_feat = torch.cat([m_proj, r_proj], dim=-1)  # [B, N, fused_dim*2]
        gate_weights = self.gate_mlp(concat_feat)          # [B, N, 2]

        # Step 3: 加权融合两个特征
        fused_feat = gate_weights[..., 0:1] * m_proj + gate_weights[..., 1:2] * r_proj  # [B, N, fused_dim]
        return fused_feat

# 加入动态版
# class MoEFusion(nn.Module):
#     def __init__(self, mamba_dim, rad_dino_dim, hidden_dim=None, fusion_dim=None):
#         super().__init__()
#         self.fusion_dim = fusion_dim or (mamba_dim + rad_dino_dim)
#         self.hidden_dim = hidden_dim or (self.fusion_dim // 2)

#         self.mamba_proj = nn.Linear(mamba_dim, self.fusion_dim)
#         self.rad_dino_proj = nn.Linear(rad_dino_dim, self.fusion_dim)

#         self.gate_mlp = nn.Sequential(
#             nn.Linear(self.fusion_dim * 2, self.hidden_dim),
#             nn.GELU(),
#             nn.Linear(self.hidden_dim, 2),
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, mamba_feat, rad_dino_feat):
#         mamba_proj_feat = self.mamba_proj(mamba_feat)
#         rad_dino_proj_feat = self.rad_dino_proj(rad_dino_feat)


#         if m_proj.shape[1] != r_proj.shape[1]:
#             target_len = min(m_proj.shape[1], r_proj.shape[1])
#             m_proj = align_tokens(m_proj, target_len)
#             r_proj = align_tokens(r_proj, target_len)


#         concat_feat = torch.cat([mamba_proj_feat, rad_dino_proj_feat], dim=-1)

#         gate_weights = self.gate_mlp(concat_feat)

#         fused_feat = gate_weights[..., 0:1] * mamba_proj_feat + gate_weights[..., 1:2] * rad_dino_proj_feat
#         return fused_feat



    
    

