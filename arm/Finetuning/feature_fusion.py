
import torch
import torch.nn as nn
from torch.nn import functional as F



class FeatureFusion(nn.Module):
    def __init__(self, mamba_dim=1024, rad_dino_dim=768, gate_hidden=None):
        super().__init__()
        in_ch = mamba_dim + rad_dino_dim
        hidden = gate_hidden or in_ch
        self.align_dino = nn.Conv2d(rad_dino_dim, mamba_dim, 1)

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1)
        )

    def forward(self, mamba_map, rad_dino_map):
        if mamba_map.shape[2:] != rad_dino_map.shape[2:]:
            rad_dino_map = F.interpolate(rad_dino_map, size=mamba_map.shape[2:], mode='bilinear', align_corners=False)

        rad_dino_map_aligned = self.align_dino(rad_dino_map)  

        combined = torch.cat([mamba_map, rad_dino_map], dim=1)
        gate = torch.sigmoid(self.fusion_gate(combined))  

        fused_feat = gate * mamba_map + (1 - gate) * rad_dino_map
        return fused_feat
