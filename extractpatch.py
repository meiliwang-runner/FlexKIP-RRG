# import os
# import torch
# import cv2
# from torchvision.io import read_image
# from tqdm import tqdm

# # 路径配置，改成你自己的路径
# IMAGE_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/daasets/mimic/image_224"
# CAM_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5/cams"
# BOX_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5/boxes"
# OUTPUT_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5/sip_features"

# TOP_K = 5
# PATCH_SIZE = 40

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def load_image(image_id):
#     # 找对应图像文件，假设是 jpg 格式
#     img_path_jpg = os.path.join(IMAGE_DIR, image_id + ".jpg")
#     img_path_png = os.path.join(IMAGE_DIR, image_id + ".png")

#     if os.path.exists(img_path_jpg):
#         img = read_image(img_path_jpg).float()  # [C,H,W]
#     elif os.path.exists(img_path_png):
#         img = read_image(img_path_png).float()
#     else:
#         return None

#     # 转单通道灰度，如果是3通道，转均值
#     if img.shape[0] == 3:
#         img = img.mean(dim=0, keepdim=True)
#     return img  # [1,H,W]

# def crop_and_resize(img_tensor, box, patch_size=PATCH_SIZE):
#     _, H, W = img_tensor.shape
#     x1, y1, x2, y2 = map(int, box)

#     # 防止越界
#     x1 = max(0, min(x1, W-1))
#     x2 = max(0, min(x2, W-1))
#     y1 = max(0, min(y1, H-1))
#     y2 = max(0, min(y2, H-1))

#     if x2 <= x1 or y2 <= y1:
#         return torch.zeros((1, patch_size, patch_size), dtype=torch.float32)

#     crop_img = img_tensor[:, y1:y2, x1:x2].squeeze(0).cpu().numpy()  # [H,W]二维数组

#     resized_img = cv2.resize(crop_img, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

#     # 转为Tensor并扩展通道维度为 [1, patch_size, patch_size]
#     patch_tensor = torch.from_numpy(resized_img).unsqueeze(0).contiguous().float()

#     return patch_tensor


# def main():
#     all_fnames = [f for f in os.listdir(CAM_DIR) if f.endswith("_cam.pt")]
#     print(f"找到 {len(all_fnames)} 张图的 CAM 文件")

#     for fname in tqdm(all_fnames):
#         image_id = fname.replace("_cam.pt", "")
#         cam_path = os.path.join(CAM_DIR, fname)
#         box_fname = fname.replace("_cam.pt", "_box.pt")
#         box_path = os.path.join(BOX_DIR, box_fname)

#         if not os.path.exists(box_path):
#             print(f"缺失 box 文件，跳过: {box_path}")
#             continue

#         img_tensor = load_image(image_id)
#         if img_tensor is None:
#             print(f"缺失图像文件，跳过: {image_id}")
#             continue

#         cam_data = torch.load(cam_path)
#         box_data = torch.load(box_path)

#         patch_dict = {}

#         # 只遍历 boxes，因为你感兴趣的是这些区域
#         for cls, boxes in box_data.items():
#             if cls not in cam_data:
#                 continue

#             # 取前 TOP_K 个 box
#             selected_boxes = boxes[:TOP_K]

#             patches = []
#             for box in selected_boxes:
#                 patch = crop_and_resize(img_tensor, box, PATCH_SIZE)  # [1,40,40]
#                 patches.append(patch)

#             if len(patches) > 0:
#                 patch_tensor = torch.stack(patches, dim=0)  # [K,1,40,40]
#                 patch_dict[cls] = patch_tensor

#         if len(patch_dict) == 0:
#             print(f"{image_id} 无有效类别区域，跳过保存")
#             continue

#         save_path = os.path.join(OUTPUT_DIR, f"{image_id}_patches.pt")
#         torch.save(patch_dict, save_path)

#     print("✅ 所有图像局部 Patch 特征提取完成")

# if __name__ == "__main__":
#     main()

import os
import torch
import cv2
from torchvision.io import read_image
from tqdm import tqdm
import concurrent.futures

# # 路径配置，改成你自己的路径
# IMAGE_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/daasets/mimic/image_224"
# CAM_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5/cams"
# BOX_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5/boxes"
# OUTPUT_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_top5/sip_features"

# 路径配置，改成你自己的路径
IMAGE_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/daasets/mimic/image_224"
CAM_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_t3z0.7_val/cams"
BOX_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_t3z0.7_val/boxes"
OUTPUT_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/region_t3z0.7_val/sip_features"

# TOP_K = 5
TOP_K = 3
PATCH_SIZE = 40
MAX_WORKERS = 8  # 并行线程数，可调节

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(image_id):
    img_path_jpg = os.path.join(IMAGE_DIR, image_id + ".jpg")
    img_path_png = os.path.join(IMAGE_DIR, image_id + ".png")

    if os.path.exists(img_path_jpg):
        img = read_image(img_path_jpg).float()  # [C,H,W]
    elif os.path.exists(img_path_png):
        img = read_image(img_path_png).float()
    else:
        return None

    if img.shape[0] == 3:
        img = img.mean(dim=0, keepdim=True)
    return img  # [1,H,W]


def crop_and_resize(img_tensor, box, patch_size=PATCH_SIZE):
    _, H, W = img_tensor.shape
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))

    if x2 <= x1 or y2 <= y1:
        return torch.zeros((1, patch_size, patch_size), dtype=torch.float32)

    crop_img = img_tensor[:, y1:y2, x1:x2].squeeze(0).cpu().numpy()
    resized_img = cv2.resize(crop_img, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    patch_tensor = torch.from_numpy(resized_img).unsqueeze(0).contiguous().float()
    return patch_tensor


def process_one_file(fname):
    image_id = fname.replace("_cam.pt", "")
    cam_path = os.path.join(CAM_DIR, fname)
    box_fname = fname.replace("_cam.pt", "_box.pt")
    box_path = os.path.join(BOX_DIR, box_fname)

    if not os.path.exists(box_path):
        # print(f"缺失 box 文件，跳过: {box_path}")
        return None

    img_tensor = load_image(image_id)
    if img_tensor is None:
        # print(f"缺失图像文件，跳过: {image_id}")
        return None

    cam_data = torch.load(cam_path)
    box_data = torch.load(box_path)

    patch_dict = {}

    for cls, boxes in box_data.items():
        if cls not in cam_data:
            continue

        selected_boxes = boxes[:TOP_K]
        patches = []
        for box in selected_boxes:
            patch = crop_and_resize(img_tensor, box, PATCH_SIZE)
            patches.append(patch)

        if patches:
            patch_tensor = torch.stack(patches, dim=0)  # [K,1,40,40]
            patch_dict[cls] = patch_tensor

    if not patch_dict:
        # print(f"{image_id} 无有效类别区域，跳过保存")
        return None

    save_path = os.path.join(OUTPUT_DIR, f"{image_id}_patches.pt")
    torch.save(patch_dict, save_path)
    return save_path


def main():
    all_fnames = [f for f in os.listdir(CAM_DIR) if f.endswith("_cam.pt")]
    print(f"找到 {len(all_fnames)} 张图的 CAM 文件")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_one_file, all_fnames), total=len(all_fnames)))

    print("✅ 所有图像局部 Patch 特征提取完成")


if __name__ == "__main__":
    main()
