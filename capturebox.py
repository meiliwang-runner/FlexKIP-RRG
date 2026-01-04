import os
import torch
from tqdm import tqdm

# 路径设置
input_dir = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5/boxes"     # 原始框文件夹
output_dir = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5_capture/boxes"   # 处理后框保存目录
os.makedirs(output_dir, exist_ok=True)

# 每个类别最多保留的 box 数
max_keep = 3

# 遍历所有 .pt 文件
for fname in tqdm(os.listdir(input_dir), desc="Processing Boxes"):
    if not fname.endswith(".pt"):
        continue

    fpath = os.path.join(input_dir, fname)
    save_path = os.path.join(output_dir, fname)

    # 加载 .pt 文件（应为字典，每个键是类名，对应一个 List[List[float]]）
    try:
        data = torch.load(fpath)
    except Exception as e:
        print(f"❌ 加载失败: {fname}，原因: {e}")
        continue

    # 截取每个类的前 max_keep 个框
    new_data = {}
    for class_name, box_list in data.items():
        if not isinstance(box_list, list):
            print(f"⚠️ 类别 {class_name} 的数据不是 list，跳过")
            continue
        new_data[class_name] = box_list[:max_keep]

    # 保存新的 .pt 文件
    torch.save(new_data, save_path)
