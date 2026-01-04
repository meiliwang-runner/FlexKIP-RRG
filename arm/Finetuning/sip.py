# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class PrototypeSoftMatcherWithCheXbert(nn.Module):
#     def __init__(self, num_classes, region_feat_dim, proto_dim, prompt_dim):
#         super().__init__()
#         self.num_classes = num_classes
#         self.proto_dim = proto_dim

#         self.semantic_prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
#         self.region_to_proto_proj = nn.Linear(region_feat_dim, proto_dim)
#         self.label_to_proto_proj = nn.Linear(proto_dim, proto_dim)

#         self.proto_to_prompt_proj = nn.Linear(proto_dim, prompt_dim)
#         self.region_to_prompt_proj = nn.Linear(region_feat_dim, prompt_dim)

#     def forward(self, region_feats, chexbert_labels):
#         """
#         region_feats: [B, N, region_feat_dim]
#         chexbert_labels: [B, N, num_classes]
#         return: [B, prompt_dim]
#         """
#         B, N, _ = region_feats.shape

#         label_semantic_prior = torch.matmul(chexbert_labels, self.semantic_prototypes)
#         label_semantic_prior = self.label_to_proto_proj(label_semantic_prior)

#         region_proj = self.region_to_proto_proj(region_feats)
#         region_norm = F.normalize(region_proj, dim=-1)
#         proto_norm = F.normalize(self.semantic_prototypes, dim=-1)
#         sim = torch.matmul(region_norm, proto_norm.t())  # [B, N, K]
#         soft_weights = F.softmax(sim, dim=-1)
#         soft_semantic = torch.matmul(soft_weights, self.semantic_prototypes)

#         fused_semantic = 0.5 * label_semantic_prior + 0.5 * soft_semantic
#         prompt_from_semantic = self.proto_to_prompt_proj(fused_semantic)
#         prompt_from_region = self.region_to_prompt_proj(region_feats)

#         prompt_region = 0.5 * prompt_from_semantic + 0.5 * prompt_from_region
#         prompt_final = prompt_region.mean(dim=1)
#         return prompt_final  # [B, prompt_dim]


# class VisualPromptDecoder(nn.Module):
#     def __init__(self, region_feat_dim, hidden_dim=768, num_layers=2, num_heads=8):
#         super().__init__()
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim * 4,
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
#         self.visual_proj = nn.Linear(region_feat_dim, hidden_dim)
#         self.learned_prompt_input = nn.Parameter(torch.randn(1, 1, hidden_dim))

#     def forward(self, region_feats):
#         B = region_feats.size(0)
#         memory = self.visual_proj(region_feats)
#         tgt = self.learned_prompt_input.expand(B, -1, -1)
#         out = self.decoder(tgt=tgt, memory=memory)
#         return out.squeeze(1)  # [B, hidden_dim]


# class SIPPromptModule(nn.Module):
#     def __init__(self,
#                  prompt_length=5,
#                  embedding_dim=768,
#                  num_classes=3,
#                  region_feat_dim=512,
#                  proto_dim=256):
#         super().__init__()

#         self.prompt_length = prompt_length
#         self.embedding_dim = embedding_dim

#         # 固定 soft prompt 初始化向量
#         self.soft_prompt = nn.Parameter(torch.randn(prompt_length, embedding_dim) * 0.02)

#         # 类别语义偏置（3类）
#         self.semantic_bias = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.02)

#         # 模块初始化
#         self.prototype_matcher = PrototypeSoftMatcherWithCheXbert(
#             num_classes=num_classes,
#             region_feat_dim=region_feat_dim,
#             proto_dim=proto_dim,
#             prompt_dim=embedding_dim
#         )
#         self.visual_prompt_decoder = VisualPromptDecoder(region_feat_dim, hidden_dim=embedding_dim)

#     def forward(self, region_feats, chexbert_labels, class_idx):
#         """
#         region_feats: [B, N, region_feat_dim]
#         chexbert_labels: [B, N, num_classes]
#         class_idx: [B] 主语义类别索引（0-病灶，1-结构，2-属性）
#         """
#         B = region_feats.size(0)

#         # [B, prompt_dim]
#         proto_prompt = self.prototype_matcher(region_feats, chexbert_labels)
#         visual_prompt = self.visual_prompt_decoder(region_feats)

#         # [B, prompt_dim]
#         fused_prompt = proto_prompt + visual_prompt

#         # [B, prompt_len, prompt_dim]
#         soft_prompt = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)

#         # 加入语义偏置
#         bias_vec = self.semantic_bias[class_idx]                  # [B, D]
#         bias_expand = bias_vec.unsqueeze(1).expand(-1, self.prompt_length, -1)  # [B, P, D]

#         fused_prompt_expand = fused_prompt.unsqueeze(1).expand(-1, self.prompt_length, -1)

#         # 🔥 直接加法融合
#         final_prompt = soft_prompt + fused_prompt_expand + bias_expand  # [B, P, D]

#         return final_prompt

# import torch.nn as nn

# class SIPPromptModule(nn.Module):
#     def __init__(self, vision_dim=1024, prompt_dim=768, num_prompt_tokens=1, num_classes=14, region_embed_dim=256):
#         super().__init__()
#         self.box_encoder = nn.Linear(4, region_embed_dim)
#         self.label_embedding = nn.Embedding(num_classes, region_embed_dim)
#         self.fusion = nn.Linear(vision_dim + 2 * region_embed_dim, prompt_dim)
#         self.output_proj = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(prompt_dim, prompt_dim)
#         )

#     def forward(self, image_features, boxes, labels):
#         box_feat = self.box_encoder(boxes)                 # [B, R, D']
#         label_feat = self.label_embedding(labels)          # [B, R, D']
#         region_feat = torch.cat([box_feat, label_feat], dim=-1)  # [B, R, 2D']
#         region_feat = region_feat.mean(dim=1)              # [B, 2D']

#         img_feat = image_features.mean(dim=1)              # [B, D]
#         fused = torch.cat([img_feat, region_feat], dim=-1) # [B, D + 2D']
#         prompt_embed = self.output_proj(self.fusion(fused))  # [B, prompt_dim]
#         return prompt_embed.unsqueeze(1)                   # [B, 1, prompt_dim]

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
        # 调试
        # print(f"[DEBUG] SIPPromptModule init: embed_dim={embed_dim} (type={type(embed_dim)})")
        # 调试

        self.embed_dim = embed_dim
        self.prompt_length = prompt_length

        # 区域编码器
        self.region_encoder = SIPRegionPromptEncoder(in_channels, patch_size, embed_dim, with_box)

        # 全局视觉提示编码 MLP（输入假设为 [B, C, H, W]）
        self.global_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 类别语义偏置向量
        self.semantic_bias = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)

        # 基础soft prompt token序列
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, embed_dim) * 0.02)

        # 多源融合权重（global, region, semantic bias）
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        # 归一化温度参数，用于区域-原型相似度计算
        self.temperature = temperature

    def prototype_attention(self, region_feats, prototype_vectors):
        """
        计算区域特征与语义原型的加权融合提示向量
        Args:
            region_feats: [B, K, embed_dim]
            prototype_vectors: [num_classes, embed_dim]
        Returns:
            region_prompt: [B, embed_dim]
        """
        # 归一化
        region_feats_norm = F.normalize(region_feats, dim=-1)   # [B, K, D]
        proto_norm = F.normalize(prototype_vectors, dim=-1)     # [num_classes, D]

        # 计算相似度
        attn_scores = torch.matmul(region_feats_norm, proto_norm.T) / self.temperature  # [B, K, num_classes]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, K, num_classes]

        # 加权求和得到每个区域对应的语义原型特征
        guided_proto_feats = torch.matmul(attn_weights, prototype_vectors)  # [B, K, embed_dim]

        # 对所有区域求平均作为区域融合提示
        region_prompt = guided_proto_feats.mean(dim=1)  # [B, embed_dim]

        return region_prompt

    def forward(self, patch_tensor, prototype_vectors, global_visual_feat, class_idx, box_tensor=None):
        """
        Args:
            patch_tensor: [B, K, 1, 40, 40] - 局部patch输入
            prototype_vectors: [num_classes, embed_dim] - 语义原型向量
            global_visual_feat: [B, in_channels, H, W] - 全局视觉特征图
            class_idx: [B] - 每张图的类别索引
            box_tensor: [B, K, 4] (optional) - 位置框
        Returns:
            final_prompt: [B, prompt_length, embed_dim] - 多token软提示
        """
        B = patch_tensor.shape[0]

        # 1. 编码区域patch特征
        region_feats = self.region_encoder(patch_tensor, box_tensor)  # [B, K, embed_dim]

        # 2. 全局视觉提示
        global_prompt = self.global_mlp(global_visual_feat)  # [B, embed_dim]

        # 3. 区域提示与语义原型匹配融合
        region_prompt = self.prototype_attention(region_feats, prototype_vectors)  # [B, embed_dim]

        # 4. 类别语义偏置
        semantic_bias_vec = self.semantic_bias[class_idx]  # [B, embed_dim]
        semantic_global = semantic_bias_vec.mean(dim=1)

        # # 5. 多源融合权重归一化
        # weights = F.softmax(self.fusion_weights, dim=0)  # [3]

        # # 6. 融合三路提示
        # fused_prompt = (
        #     weights[0] * global_prompt +
        #     weights[1] * region_prompt +
        #     weights[2] * semantic_bias_vec
        # )  # [B, embed_dim]

        # # 7. 基础软提示token expand
        # soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)  # [B, prompt_length, embed_dim]

        # # 8. 融合提示 expand 成多token，加到基础软提示上
        # fused_prompt_expanded = fused_prompt.unsqueeze(1).expand(-1, self.prompt_length, -1)  # [B, prompt_length, embed_dim]

        # final_prompt = soft_prompt_expanded + fused_prompt_expanded  # [B, prompt_length, embed_dim]

        # 5. 多源融合权重归一化
        weights = F.softmax(self.fusion_weights, dim=0)  # [3]
        print("fusion_weights:", weights)  # 打印融合权重

        # 6. 融合三路提示
        # print("global_prompt shape:", global_prompt.shape)         # [B, embed_dim]
        # print("region_prompt shape:", region_prompt.shape)         # [B, embed_dim]
        # print("semantic_bias_vec shape:", semantic_bias_vec.shape) # [B, embed_dim]
        # print("semantic_global shape:", semantic_global.shape)

        # fused_prompt = (
        #     weights[0] * global_prompt +
        #     weights[1] * region_prompt +
        #     weights[2] * semantic_bias_vec
        # )  # [B, embed_dim]
        fused_prompt = (
            weights[0] * global_prompt +
            weights[1] * region_prompt +
            weights[2] * semantic_global
        )  # [B, embed_dim]
        

        # print("fused_prompt shape:", fused_prompt.shape)

        # 7. 基础软提示token expand
        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(B, -1, -1)  # [B, prompt_length, embed_dim]
        # print("soft_prompt_expanded shape:", soft_prompt_expanded.shape)

        # 8. 融合提示 expand 成多token，加到基础软提示上
        fused_prompt_expanded = fused_prompt.unsqueeze(1).expand(-1, self.prompt_length, -1)  # [B, prompt_length, embed_dim]
        # print("fused_prompt_expanded shape:", fused_prompt_expanded.shape)

        final_prompt = soft_prompt_expanded + fused_prompt_expanded  # [B, prompt_length, embed_dim]
        # print("final_prompt shape:", final_prompt.shape)
        
        return final_prompt
        

