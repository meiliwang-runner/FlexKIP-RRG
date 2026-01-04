import os
import math  # 添加此行
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
import torch.nn as nn
import numpy as np
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.nn import functional as F
from arm.Finetuning.models_mamba import arm_base_pz16, arm_large_pz16
from arm.Finetuning.util.pos_embed import interpolate_pos_embed


from arm.Finetuning.feature_fusion import FeatureFusion  # 需要自己新建这个模块文件（前面咱俩写过的FeatureFusion类）

class MambaXrayVLCLIP(pl.LightningModule):
    """
    MambaXrayVLCLIP model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.text_encoder_type = args.text_encoder_type


        # 视觉分支特征
        # 加载rad_dino模型
        # 这部分直接加载模型进来不进行微调
        print(f'Loading assistant_vision encoder:/media/wangyujie/CXPMRG_Bench_MambaXray_VL/huggingface/microsoft/rad-dino')
        self.rad_dino = AutoModel.from_pretrained("/media/wangyujie/CXPMRG_Bench_MambaXray_VL/huggingface/microsoft/rad-dino")
        # processor = AutoImageProcessor.from_pretrained("/media/wangyujie/CXPMRG_Bench_MambaXray_VL/huggingface/microsoft/rad-dino")
        # for name, param in self.rad_dino.named_parameters():
        #     param.requires_grad = False  # 冻结全部
        # # 可以只解冻 rad-dino最后几层，比如
        # for name, param in self.rad_dino.named_parameters():
        #     if 'layernorm' in name or 'blocks.11' in name:  # 举例
        #         param.requires_grad = True
        print(22222222222)
        # 视觉分支特征

        # 混合模块参数初始化
        self.dino_proj = nn.Linear(768, 1024)  # 将 dino_feat 的通道数从 768 调整为 1024

        print(f'Loading main_vision encoder:{args.vision_model}')
        print(f'your mamba vision encoder model {args.type}')
        # print(f'you choose the vison encoder model Mamba {self.type}')
        if args.type == 'base':
            self.visual_encoder = arm_base_pz16(args.type)
        else:
            self.visual_encoder = arm_large_pz16(args.type)
        
        # # 打印视觉编码器的所有模块名称
        # print("Visual Encoder Modules:")
        # for name, module in self.visual_encoder.named_modules():
        #     print(name)


        finetune = args.vision_model
        # 查看模型是否走到了这一步
        # print(finetune)
        
        # 冻结大部分层，仅微调最后几层
        for name, param in self.visual_encoder.named_parameters():
            if "layers.20" in name or "layers.21" in name or "layers.22" in name or "layers.23" in name:
                param.requires_grad = True  # 微调最后几层
                # print("lora")
            else:
                param.requires_grad = False  # 冻结其他层
                # print("freeze")

        if finetune!='None':
            checkpoint = torch.load(finetune, map_location='cpu')
            
            # print(f"Checkpoint keys: {checkpoint.keys()}")

            print(f"Load arm pre-trained checkpoint from: {finetune}" )
            
            checkpoint_model = checkpoint['model']
            # if checkpoint_model is None:
            #     raise ValueError(f"Checkpoint does not contain 'model' key. Check the file: {finetune}")

            new_dict = {}
            for k, v in checkpoint_model.items():
                if "conv1d" in k:
                    new_dict[k.replace("conv1d", "conv1d_b")] = v
                    new_dict[k.replace("conv1d", "conv1d_c")] = v
                    new_dict[k.replace("conv1d", "conv1d_c_b")] = v
                if "dt_proj" in k:
                    new_dict[k.replace("dt_proj", "dt_proj_b")] = v
                    new_dict[k.replace("dt_proj", "dt_proj_c")] = v
                    new_dict[k.replace("dt_proj", "dt_proj_c_b")] = v
                if "x_proj" in k:
                    new_dict[k.replace("x_proj", "x_proj_b")] = v
                    new_dict[k.replace("x_proj", "x_proj_c")] = v
                    new_dict[k.replace("x_proj", "x_proj_c_b")] = v
                if "A" in k:
                    new_dict[k.replace("A", "A_b")] = v
                    new_dict[k.replace("A", "A_c")] = v
                    new_dict[k.replace("A", "A_c_b")] = v
                if "D" in k:
                    new_dict[k.replace("D", "D_b")] = v
                    new_dict[k.replace("D", "D_c")] = v
                    new_dict[k.replace("D", "D_c_b")] = v
                if "dec" not in k:
                    new_dict[k] = v

            # interpolate position embedding
            new_dict = interpolate_pos_embed(self.visual_encoder, new_dict)

            # load pre-trained model
            self.visual_encoder.load_state_dict(new_dict, strict=False)
            # print(1111111111111)
        
        if args.vis_use_lora:
            # peft_config_visual = LoraConfig(
            #                         r=args.vis_r,
            #                         lora_alpha=args.vis_alpha,
            #                         target_modules=["query", "value"],
            #                         lora_dropout=args.lora_dropout,
            #                         bias="none",
            #                         modules_to_save=["classifier"],
            #                     )
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["x_proj", "dt_proj", "w1", "w2", "w3"],  # 修改为实际存在的模块
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )

            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print(f"Loading text encoder : {self.text_encoder_type}...")
        # if self.text_encoder_type == 'Bio_ClinicalBERT':  
        #     self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        #     if self.tokenizer.bos_token_id is None:
        #         self.tokenizer.bos_token_id = self.tokenizer.cls_token_id
        #     self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")    
        if self.text_encoder_type == 'Bio_ClinicalBERT':  
            self.tokenizer = AutoTokenizer.from_pretrained("/media/wangyujie/CXPMRG_Bench_MambaXray_VL/huggingface/emilyalsentzer/Bio_ClinicalBERT")
            if self.tokenizer.bos_token_id is None:
                self.tokenizer.bos_token_id = self.tokenizer.cls_token_id
            self.text_encoder = AutoModel.from_pretrained("/media/wangyujie/CXPMRG_Bench_MambaXray_VL/huggingface/emilyalsentzer/Bio_ClinicalBERT")    
            
        print(f"Loading text encoder : {self.text_encoder_type}done")


        # 特征融合模块--添加
        self.fusion_module = FeatureFusion(mamba_dim=self.visual_encoder.num_features, rad_dino_dim=self.rad_dino.config.hidden_size)
        # 特征融合模块--添加

        self.projection_dim = args.projection_dim
        self.vision_proj = nn.Linear(self.visual_encoder.num_features, self.projection_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.projection_dim)
        self.temperature = 0.07 # 0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))

        self.min_loss = -1

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    # def encode_img(self, images):
    #     image_embeds = []
    #     for image in images:
    #         device = image.device
    #         image_embed = self.visual_encoder(image)
    #         image_embeds.append(image_embed)
            
    #     image_embeds = torch.stack(image_embeds).mean(0)
    #     image_embeds = image_embeds.mean(dim=1)
    #     image_embeds = self.vision_proj(image_embeds)
    #     return image_embeds
    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            mamba_feat = self.visual_encoder(image)  # (B, C, H, W)
            dino_feat = self.rad_dino(image).last_hidden_state  # (B, N, C)，需要 reshape
            print(f"mamba_feat.shape: {mamba_feat.shape}")
            print(f"dino_feat.shape: {dino_feat.shape}")
            # # mamba_feat.shape: torch.Size([48, 197, 1024])
            # # dino_feat.shape: torch.Size([48, 257, 768])
            
            # # 处理dino_feat，reshape成(B, C, H, W)
            # B, N, C = dino_feat.shape
            # # H = int(math.sqrt(N))
            # # W = N // H
            # # if H * W != N:
            # #     raise ValueError(f"N={N} 无法 reshape 为 [B, C, H, W]")
            # H = W = int(math.sqrt(N))
            # print(f"Calculated H: {H}, W: {W}")
            # dino_feat = dino_feat.permute(0,2,1).reshape(B, C, H, W)
            # 简洁高效，避免插值带来的开销和不确定性
            # 相比插值（如 F.interpolate），直接截断序列长度更快、更稳定；
            # 适合训练时处理大 batch 或长序列场景，节省显存和计算时间；
            # 对于 dino_feat，其 token 是 patch-wise 的，截断前面的 197 个不会严重损失语义（ViT 通常前几个 token 已涵盖主区域信息）。
            # ② 使用线性层对齐通道维度是标准做法
            # nn.Linear(768 → 1024) 是最自然的变换方式，不引入空间感知偏置；
            # 1×1 卷积虽然也可用，但更适用于带有空间结构的特征图（如 CNN 输出），你这里的 dino_feat 是 Transformer 输出，线性更合适。
            # ③ 保持对齐一致性，简化后续融合
            # 你将两个特征都变成 [B, 197, 1024]，非常适合用于加法、拼接或 cross-attention 融合；
            # fusion_module 的设计可以专注于融合逻辑，无需再处理维度兼容性问题。
            # ④ 使用两次 mean() 做空间与 batch 归约，稳健输出单一图像嵌入

            # # 对齐序列长度
            # dino_feat = dino_feat[:, :mamba_feat.shape[1], :]  # 截断到 197

            # # 对齐通道数
            # dino_feat = self.dino_proj(dino_feat)  # (B, 197, 1024)

            # print(f"dino_feat.shape: {dino_feat.shape}")
            # dino_feat.shape: torch.Size([48, 197, 1024])

            # 应用到你的两个分支
            mamba_map = self.tokens_to_map(mamba_feat)      # [B, 1024, 14, 14]
            dino_map = self.tokens_to_map(dino_feat)   # [B, 768, 16, 16]
            print(f"mamba_map.shape: {mamba_map.shape}")
            print(f"dino_map.shape: {dino_map.shape}")

            # 融合
            fused_feat = self.fusion_module(mamba_map, dino_map)
            print(f"fused_feat shape: {fused_feat.shape}")

            # 后处理
            image_embeds.append(fused_feat)
            # print(f"image_embeds shape: {image_embeds.shape}")

        image_embeds = torch.stack(image_embeds).mean(0)  # batch平均
        print(f"image_embeds shape after stacking and averaging: {image_embeds.shape}")
        image_embeds = image_embeds.mean(dim=1)  # spatial平均
        print(f"image_embeds shape after spatial averaging: {image_embeds.shape}")

        # 这表明 image_embeds 的形状为 (672, 14)，而 self.vision_proj 的权重矩阵形状为 (1024, 2048)，
        # 两者的形状不匹配，无法进行矩阵乘法。
        image_embeds = self.vision_proj(image_embeds)
        print(f"image_embeds shape after vision_proj: {image_embeds.shape}")
        return image_embeds


    # def tokens_to_map(token_feat):
    #     """
    #     token_feat: [B, N, C]  （含 CLS）
    #     returns:   [B, C, P, P]
    #     针对新增加的分支设计的reshape
    #     """
    #     B, N, C = token_feat.shape

    #     # 1) 去掉 CLS token
    #     feat = token_feat[:, 1:, :]                # [B, N-1, C]

    #     # 2) 计算 P
    #     M = feat.shape[1]                          # M = N - 1
    #     P = int(math.sqrt(M))
    #     assert P * P == M, f"非平方 token 数：{M}"

    #     # 3) 从 [B, M, C] → [B, C, P, P]
    #     feat = feat.permute(0, 2, 1)               # [B, C, M]
    #     feat = feat.view(B, C, P, P)               # [B, C, P, P]
    #     return feat
    @staticmethod
    def tokens_to_map(token_feat):
        """
        token_feat: [B, N, C]  （含 CLS）
        returns:   [B, C, P, P]
        针对新增加的分支设计的reshape
        """
        B, N, C = token_feat.shape

        # 1) 去掉 CLS token
        feat = token_feat[:, 1:, :]                # [B, N-1, C]

        # 2) 计算 P
        M = feat.shape[1]                          # M = N - 1
        P = int(math.sqrt(M))
        assert P * P == M, f"非平方 token 数：{M}"

        # 3) 从 [B, M, C] → [B, C, P, P]
        feat = feat.permute(0, 2, 1)               # [B, C, M]
        feat = feat.view(B, C, P, P)               # [B, C, P, P]
        return feat

    

    def encode_txt(self, text_tokens):
        if self.text_encoder_type == 'Bio_ClinicalBERT':
            text_features = self.text_encoder(text_tokens['input_ids'], attention_mask = text_tokens['attention_mask'])["last_hidden_state"]
        eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
        text_features = text_features[torch.arange(text_features.shape[0]), eos_token_indices]
        text_features = self.text_proj(text_features)
        return text_features

    def forward(self, samples):
        image = samples["image"]
        report = samples["input_text"]
        text_tokens = self.tokenizer(report, padding="max_length", truncation=True, return_tensors="pt", max_length=128).to(image[0].device)
        image_features = self.encode_img(image)
        text_features = self.encode_txt(text_tokens)
        
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        # breakpoint()
        labels = torch.arange(logits_per_image.shape[0],dtype=torch.long,device=logits_per_image.device)

        # # loss = self.cliploss(image_features, text_features, self.logit_scale.exp())["contrastive_loss"]
        # loss = (
        #     F.cross_entropy(logits_per_image, labels) +
        #     F.cross_entropy(logits_per_text, labels)
        # ) / 2
        # return {"loss": loss}

        # 特征一致性正则项
        original_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        consistency_loss = F.mse_loss(mamba_feat, rad_dino_feat.detach())
        total_loss = original_loss + 0.1 * consistency_loss
        # 特征一致性正则项

        return {"loss": total_loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, loss):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_loss{:3f}.pth".format(current_epoch, global_step, loss),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics["loss"]
        if self.min_loss == -1 :
            self.min_loss = avg_loss
            self.save_checkpoint(self.min_loss)
        elif avg_loss < self.min_loss and self.min_loss != -1:
            self.min_loss = avg_loss
            self.save_checkpoint(self.min_loss)
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}
    def configure_optimizers(self):
    # 只优化融合模块 + rad-dino高层可训练部分
        params = list(self.fusion_module.parameters()) + [p for p in self.rad_dino.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

    # 计算模型的参数量
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        return total_params, trainable_params