
import torch
import torch.nn as nn
import torch.nn.functional as F

class SIPRegionPromptEncoder(nn.Module):
    def __init__(self, in_channels=1, patch_size=40, embed_dim=256, with_box=True):
        super().__init__()
        self.with_box = with_box
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # → (K, 64, 1, 1)
        )
        self.patch_fc = nn.Linear(64, embed_dim)

        if with_box:
            self.box_fc = nn.Linear(4, embed_dim)

        self.fusion_fc = nn.Linear(embed_dim * (2 if with_box else 1), embed_dim)

    def forward(self, patch_tensor, box_tensor=None):
        """
        Args:
            patch_tensor: [B, K, 1, 40, 40]
            box_tensor: [B, K, 4] or None
        Returns:
            region_feats: [B, K, embed_dim]
        """
        

        B, K, C, H, W = patch_tensor.shape
        patch_tensor = patch_tensor.view(B * K, C, H, W)  # 合并batch和patch维度

        patch_feat = self.encoder(patch_tensor)  # [B*K, 64, 1, 1]
        patch_feat = patch_feat.view(B * K, -1)  # [B*K, 64]
        patch_feat = self.patch_fc(patch_feat)   # [B*K, embed_dim]

        if self.with_box and box_tensor is not None:
            box_tensor = box_tensor.view(B * K, 4).float()
            box_feat = self.box_fc(box_tensor)  # [B*K, embed_dim]
            feat = torch.cat([patch_feat, box_feat], dim=-1)  # [B*K, 2*embed_dim]
        else:
            feat = patch_feat

        fused = self.fusion_fc(feat)  # [B*K, embed_dim]
        region_feats = fused.view(B, K, self.embed_dim)
        return region_feats


class SIPPromptModule(nn.Module):
    def __init__(self,
                 in_channels=1,
                 patch_size=40,
                 embed_dim=256,
                 with_box=True,
                 num_classes=10,
                 prompt_length=8,
                 temperature=1.0):
        super().__init__()
        

        self.embed_dim = embed_dim
        self.prompt_length = prompt_length

        
        self.region_encoder = SIPRegionPromptEncoder(in_channels, patch_size, embed_dim, with_box)

        
        self.global_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

       
        self.semantic_bias = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)

       
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)

        
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        
        self.temperature = temperature

    def prototype_attention(self, region_feats, prototype_vectors):
        """
        Args:
            region_feats: [B, K, embed_dim]
            prototype_vectors: [num_classes, embed_dim]
        Returns:
            region_prompt: [B, embed_dim]
        """
        region_feats_norm = F.normalize(region_feats, dim=-1)   
        proto_norm = F.normalize(prototype_vectors, dim=-1)    

        attn_scores = torch.matmul(region_feats_norm, proto_norm.T) / self.temperature  
        attn_weights = F.softmax(attn_scores, dim=-1)  

        guided_proto_feats = torch.matmul(attn_weights, prototype_vectors) 

        region_prompt = guided_proto_feats.mean(dim=1)  

        return region_prompt

    def forward(self, patch_tensor, prototype_vectors, global_visual_feat, class_idx, box_tensor=None):
        B = patch_tensor.shape[0]

        region_feats = self.region_encoder(patch_tensor, box_tensor)  

        global_prompt = self.global_mlp(global_visual_feat)  

        region_prompt = self.prototype_attention(region_feats, prototype_vectors) 

        semantic_bias_vec = self.semantic_bias[class_idx]  
        semantic_global = semantic_bias_vec.mean(dim=1)

        weights = F.softmax(self.fusion_weights, dim=0)  
        print("fusion_weights:", weights)  

        fused_prompt = (
            weights[0] * global_prompt +
            weights[1] * region_prompt +
            weights[2] * semantic_global
        )  
        

        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)  
       
        fused_prompt_expanded = fused_prompt.unsqueeze(1).expand(-1, self.prompt_length, -1)  

        final_prompt = soft_prompt_expanded + fused_prompt_expanded  
        
        return final_prompt
        

