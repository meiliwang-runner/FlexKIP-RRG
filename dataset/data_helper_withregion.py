import os
import json
import re
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.data as data
from transformers import AutoImageProcessor


class ChestXrayWithRegionDataset(data.Dataset):
    def __init__(self, image_dir, region_dir, annotation_json, base_dir, parser, patch_size=40):
        self.image_dir = image_dir
        self.region_dir = region_dir
        self.base_dir = base_dir
        self.parser = parser
        self.patch_size = patch_size

        self.meta = annotation_json
        print(f"Dataset loaded. Total samples: {len(self.meta)}")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        features = self.meta[idx]
        parsed = self.parser.parse(features)
        input_text = parsed['input_text']
        images = parsed['image']

        image_file = features['image_path'][0]
        image_id = os.path.splitext(image_file)[0]
        region_path = os.path.join(self.region_dir, f"{image_id}.pt")
        region_data = torch.load(region_path)
        patch_tensor = region_data.get('patch_tensor', None)
        boxes = region_data.get('box_tensor', None)
        labels = region_data.get('class_idx', None)

        return {
            'id': image_id,
            'input_text': input_text,
            'image': images,
            'patch_tensor': patch_tensor,
            'box_tensor': boxes,
            'class_idx': labels,
        }


class FieldParser:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(
            '/media/wangyujie/CXPMRG_Bench_MambaXray_VL/huggingface/microsoft/swin-base-patch4-window7-224'
        )

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt", size=self.args.input_size).pixel_values
        return pixel_values[0]  # [1, 3, 224, 224] -> [3, 224, 224]

    def clean_report(self, report):
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('1. ', '').replace('. 2. ', '. ').strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            return ' . '.join(tokens) + ' .'
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ').strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            return ' . '.join(tokens) + ' .'

    def parse(self, features):
        if self.dataset == "chinese":
            report = features.get("image_finding", "")
        else:
            report = features.get("report", "")
        report = self.clean_report(report)
        image_path = features['image_path'][0]
        with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
            arr = np.array(pil)
            if len(arr.shape) != 3 or arr.shape[-1] != 3:
                arr = np.array(pil.convert("RGB"))
            image_tensor = self._parse_image(arr)
        return {"input_text": report, "image": image_tensor}


def custom_collate_fn(batch):
    batch_out = {}
    batch_out['id'] = [item['id'] for item in batch]
    batch_out['input_text'] = [item['input_text'] for item in batch]
    batch_out['image'] = torch.stack([item['image'] for item in batch])  # [B, 3, 224, 224]

   
    min_len_patch = min([t.shape[0] for t in [item['patch_tensor'] for item in batch]])
    min_len_box = min([t.shape[0] for t in [item['box_tensor'] for item in batch]])
    min_len_cls = min([t.shape[0] for t in [item['class_idx'] for item in batch]])

   
    patch_tensors = [item['patch_tensor'][:min_len_patch] for item in batch]
    batch_out['patch_tensor'] = torch.stack(patch_tensors)

    
    box_tensors = [item['box_tensor'][:min_len_box] for item in batch]
    batch_out['box_tensor'] = torch.stack(box_tensors)

    
    class_idxs = [item['class_idx'][:min_len_cls] for item in batch]
    batch_out['class_idx'] = torch.stack(class_idxs)

    return batch_out




def create_datasets(args):
    with open(args.annotation, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    train_ids = full_data['train']
    val_ids = full_data['val']
    test_ids = full_data['test']

    if getattr(args, 'use_sip', False):
        parser = FieldParser(args)
        train_dataset = ChestXrayWithRegionDataset(args.base_dir, args.region_root, train_ids, args.base_dir, parser, args.patch_size)
        val_dataset = ChestXrayWithRegionDataset(args.base_dir, args.region_root, val_ids, args.base_dir, parser, args.patch_size)
        test_dataset = ChestXrayWithRegionDataset(args.base_dir, args.region_root, test_ids, args.base_dir, parser, args.patch_size)
    else:
        train_dataset = ParseDataset(args, 'train')
        val_dataset = ParseDataset(args, 'val')
        test_dataset = ParseDataset(args, 'test')

    return train_dataset, val_dataset, test_dataset



