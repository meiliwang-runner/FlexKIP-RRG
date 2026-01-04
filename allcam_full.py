# 加入了nms控制,但是不使用置信度
import os
import json
import torch
import numpy as np
import cv2
from torchvision.io import read_image
from torchcam.methods import SmoothGradCAMpp
import torchxrayvision as xrv
from tqdm import tqdm
import logging
from torchvision.ops import nms

# ================= 参数配置 =================
IMAGE_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/daasets/mimic/image_224"
JSON_PATH = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/daasets/mimic/mimic+inter+intra1.json"
SAVE_DIR = "/media/wangyujie/CXPMRG_Bench_MambaXray_VL/promot/cam_results_top5"
CAM_DIR = os.path.join(SAVE_DIR, "cams")
BOX_DIR = os.path.join(SAVE_DIR, "boxes")
ALL_BOX_JSON = os.path.join(SAVE_DIR, "all_boxes.json")

BOX_SIZE = 40
TOP_K = 5

TARGET_CLASSES = [
    "Atelectasis", "Consolidation", "Pneumothorax", "Edema", "Effusion", "Pneumonia",
    "Cardiomegaly", "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"
]

# ================= 日志配置 =================
logging.basicConfig(filename=os.path.join(SAVE_DIR, "cam_processing.log"),
                    level=logging.INFO, format="%(asctime)s - %(message)s")

# ================= 工具函数 =================
def init_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.eval()
    return model

def preprocess_image(img_path):
    img = read_image(img_path).float()
    if img.shape[0] == 3:
        img = img.mean(dim=0, keepdim=True)
    return img.unsqueeze(0)


def extract_topk_boxes(cam_tensor, orig_size, k=5, box_size=40, iou_threshold=0.3):
    cam_np = cam_tensor.squeeze().detach().cpu().numpy()
    cam_norm = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

    h_cam, w_cam = cam_norm.shape
    orig_h, orig_w = orig_size

    cam_flat = cam_norm.flatten()
    topk_indices = np.argpartition(cam_flat, -k)[-k:]
    topk_indices = topk_indices[np.argsort(-cam_flat[topk_indices])]

    boxes, scores = [], []
    for idx in topk_indices:
        y_cam, x_cam = np.unravel_index(idx, cam_norm.shape)
        x_img = int(x_cam * orig_w / w_cam)
        y_img = int(y_cam * orig_h / h_cam)

        half = box_size // 2
        x1 = max(0, x_img - half)
        y1 = max(0, y_img - half)
        x2 = min(orig_w - 1, x_img + half)
        y2 = min(orig_h - 1, y_img + half)

        boxes.append([x1, y1, x2, y2])
        scores.append(cam_norm[y_cam, x_cam])

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    keep = nms(boxes_tensor, scores_tensor, iou_threshold)
    keep = keep[:k]  # 最多保留 k 个

    selected_boxes = boxes_tensor[keep].tolist()
    return selected_boxes



def process_image(image_path, model, cam_extractor, device):
    full_path = os.path.join(IMAGE_DIR, image_path)
    if not os.path.exists(full_path):
        logging.warning(f"Image not found: {full_path}")
        return {}, {}

    img_tensor = preprocess_image(full_path).to(device)
    img_tensor.requires_grad_()
    output = torch.sigmoid(model(img_tensor))
    orig_size = (img_tensor.shape[2], img_tensor.shape[3])

    cam_result = {}
    box_result = {}

    for cls in TARGET_CLASSES:
        if cls not in model.pathologies:
            continue
        idx = model.pathologies.index(cls)
        model.zero_grad()
        output[0, idx].backward(retain_graph=True)

        cam_map = cam_extractor(idx, output)
        cam = list(cam_map.values())[0] if isinstance(cam_map, dict) else cam_map[0]
        # boxes = extract_topk_boxes(cam, orig_size, k=TOP_K, box_size=BOX_SIZE)
        boxes = extract_topk_boxes(cam, orig_size, k=TOP_K, box_size=BOX_SIZE, iou_threshold=0.3)


        cam_result[cls] = cam.detach().cpu()
        box_result[cls] = boxes

    return cam_result, box_result

# ================= 主执行流程 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model().to(device)
    cam_extractor = SmoothGradCAMpp(model, target_layer="features.denseblock4.denselayer16.conv2")

    os.makedirs(CAM_DIR, exist_ok=True)
    os.makedirs(BOX_DIR, exist_ok=True)

    with open(JSON_PATH, "r") as f:
        data = json.load(f)["train"]

    all_boxes_dict = {}

    for item in tqdm(data, desc="Processing all images"):
        try:
            for img_path in item["image_path"]:
                image_id = os.path.splitext(os.path.basename(img_path))[0]

                cam_result, box_result = process_image(img_path, model, cam_extractor, device)

                # 保存 CAM 和 Box 的 .pt 文件
                torch.save(cam_result, os.path.join(CAM_DIR, f"{image_id}_cam.pt"))
                torch.save(box_result, os.path.join(BOX_DIR, f"{image_id}_box.pt"))

                all_boxes_dict[image_id] = box_result
                logging.info(f"Processed {img_path}")

        except Exception as e:
            logging.error(f"Error processing {item['id']}: {str(e)}")

    # 保存所有 box 的 JSON 汇总
    with open(ALL_BOX_JSON, "w") as jf:
        json.dump(all_boxes_dict, jf, indent=2)

    print(f"\n✅ CAM and box extraction completed. Results saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
