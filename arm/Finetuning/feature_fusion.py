
import torch
import torch.nn as nn
from torch.nn import functional as F



# class FeatureFusion(nn.Module):
#     def __init__(self, mamba_dim=512, rad_dino_dim=768):
#         super().__init__()
#         self.rad_dino_proj = nn.Conv2d(rad_dino_dim, mamba_dim, 1)
#         self.fusion_gate = nn.Sequential(
#             nn.Conv2d(mamba_dim * 2, mamba_dim, 1),
#             nn.GELU(),
#             nn.Conv2d(mamba_dim, 2, 1)  # 输出两个通道，对应两个分支的融合权重
#         )
        
#     def forward(self, mamba_feat, rad_dino_feat):
#         rad_dino_feat = F.interpolate(rad_dino_feat, size=mamba_feat.shape[2:], mode='bilinear', align_corners=False)
#         rad_dino_feat = self.rad_dino_proj(rad_dino_feat)

#         # combined = torch.cat([mamba_feat, rad_dino_feat], dim=1)
#         # gates = self.fusion_gate(combined)  # [B, 2, H, W]
#         # gates = F.softmax(gates, dim=1)

#         # fused_feat = gates[:, 0:1] * mamba_feat + gates[:, 1:2] * rad_dino_feat
class FeatureFusion(nn.Module):
    def __init__(self, mamba_dim=1024, rad_dino_dim=768, gate_hidden=None):
        super().__init__()
        in_ch = mamba_dim + rad_dino_dim
        hidden = gate_hidden or in_ch

        # 新增：对 rad_dino_map 进行升维（768 → 1024）
        self.align_dino = nn.Conv2d(rad_dino_dim, mamba_dim, 1)

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1)
        )

    def forward(self, mamba_map, rad_dino_map):
        if mamba_map.shape[2:] != rad_dino_map.shape[2:]:
            rad_dino_map = F.interpolate(rad_dino_map, size=mamba_map.shape[2:], mode='bilinear', align_corners=False)

        # 先对 rad_dino_map 通道数对齐
        rad_dino_map_aligned = self.align_dino(rad_dino_map)  # [B,1024,H,W]

        combined = torch.cat([mamba_map, rad_dino_map], dim=1)
        gate = torch.sigmoid(self.fusion_gate(combined))  # [B, 1, H, W]

        fused_feat = gate * mamba_map + (1 - gate) * rad_dino_map
        return fused_feat
