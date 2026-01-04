import torch
import torch.nn as nn
import torch.nn.functional as F


def align_tokens(x, target_len):
    B, N, C = x.shape
    x = x.permute(0, 2, 1)  # [B, C, N]
    x = F.adaptive_avg_pool1d(x, target_len)  # [B, C, target_len]
    x = x.permute(0, 2, 1)  # [B, target_len, C]
    return x

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
        m_proj = self.mamba_proj(mamba_feat)       
        r_proj = self.raddino_proj(raddino_feat)   

        if m_proj.shape[1] != r_proj.shape[1]:
            target_len = min(m_proj.shape[1], r_proj.shape[1])
            m_proj = align_tokens(m_proj, target_len)
            r_proj = align_tokens(r_proj, target_len)

        
        concat_feat = torch.cat([m_proj, r_proj], dim=-1)  
        gate_weights = self.gate_mlp(concat_feat)         

        
        fused_feat = gate_weights[..., 0:1] * m_proj + gate_weights[..., 1:2] * r_proj  
        return fused_feat


    

