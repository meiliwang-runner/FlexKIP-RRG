import os
import torch
from tqdm import tqdm

# 路径设置
input_dir = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5/sip_features"     # 原始 .pt 文件目录
output_dir = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5_capture/sip_features"   # 处理后保存目录
os.makedirs(output_dir, exist_ok=True)

# 每个类别最多保留的 patch 数
max_keep = 3

# 遍历所有 .pt 文件
for fname in tqdm(os.listdir(input_dir), desc="Processing"):
    if not fname.endswith(".pt"):
        continue

    fpath = os.path.join(input_dir, fname)
    save_path = os.path.join(output_dir, fname)

    # 加载 .pt 文件
    try:
        data = torch.load(fpath)
    except Exception as e:
        print(f"❌ 加载失败: {fname}，原因: {e}")
        continue

    # 构造新数据
    new_data = {}
    for class_name, patch_tensor in data.items():
        if not isinstance(patch_tensor, torch.Tensor):
            continue
        if patch_tensor.dim() == 4:  # [K, 1, H, W] or [K, 3, H, W]
            keep_tensor = patch_tensor[:max_keep].clone()
            new_data[class_name] = keep_tensor
        else:
            print(f"⚠️ 跳过类别 {class_name}，因为 tensor 维度为 {patch_tensor.shape}")

    # 保存新的 .pt 文件
    torch.save(new_data, save_path)
