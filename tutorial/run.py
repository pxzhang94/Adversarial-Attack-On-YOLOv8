import sys
sys.path.append("../")
import os
import cv2
import json
import math
import glob
import uuid
import shutil
from pathlib import Path
import random
import errno
from collections import defaultdict
from typing import List, Dict, Tuple, Callable, Optional
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
import yaml

from utils.config import TRANSFORM_CONFIG, TRANSFORM_FUNCTION
from utils.dataUtil import read_yolo_labels, write_yolo_labels, rotate_yolo_bboxes
from utils.matchUtil import match_and_score_single 






# model = YOLO("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/tutorial/runs/detect/train/weights/best.pt")
# model = YOLO("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/model/sutd_aidx_3/weights/best.pt")
model = YOLO("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/model/sutd_aidx/weights/best.pt")
batch_size = 32
conf = 0.25 #置信度阈值
device = '0'
classes = None 
iou_thresh = 0.5


def batch_inference(img_paths, output_label_dir):
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i + batch_size]

        # 直接传入路径列表，Ultralytics 会内部按批处理
        results = model(
            [str(p) for p in batch],
            conf=conf,
            device=device,
            classes=classes,
            verbose=False
        )

        # 对齐每张图片的结果，一一保存
        for img_path, res in zip(batch, results):
            # 优先使用xywhn(归一化中心点+宽高)，避免自己再做归一化
            # res.boxes.cls: (N,1) 类别；res.boxes.conf: 置信度；res.boxes.xywhn: (N,4)
            label_txt = output_label_dir / (img_path.stem + ".txt")
            with open(label_txt, "w", encoding="utf-8") as f:
                if res.boxes is not None and len(res.boxes) > 0:
                    clses = res.boxes.cls.cpu().numpy().astype(int).tolist()
                    xywhn = res.boxes.xywhn.cpu().numpy().tolist()
                    for c, (xc, yc, w, h) in zip(clses, xywhn):
                        f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                # 若无检测结果，将生成空txt（很多训练管线也接受）

def acc_cal(pred_folder, reference_folder, img_folder, is_xywh=True):
    image_total = 0
    image_correct = 0
    object_total = 0
    object_correct = 0
    image_wrong_path = []
    pred_paths = sorted([p for p in pred_folder.iterdir() if p.suffix.lower() == ".txt"])   
    objects_per_label = {}
    correct_per_label = {}    
    for pred_path in pred_paths:
        img_path = img_folder / f"{pred_path.stem}.jpg" 
        reference_path = reference_folder / pred_path.name
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        H, W = img.shape[:2]  # 高, 宽
        
        gts_yolo = read_yolo_labels(reference_path)
        pred_path_str = str(pred_path.absolute())
        if 'rotation' in pred_path_str:
            angle = int(pred_path_str.split('/')[-3].split('_')[-1])
            gts_yolo = rotate_yolo_bboxes(gts_yolo, angle, W, H)
        
        preds_yolo = read_yolo_labels(pred_path)
        img_ok, obj_tot, obj_ok, objs_per_label, crt_per_label = match_and_score_single(gts_yolo, preds_yolo, W, H, iou_thresh, is_xywh)
        image_total += 1
        image_correct += int(img_ok)
        object_total += obj_tot
        object_correct += obj_ok
        for k in objs_per_label.keys():
            objects_per_label[k] = objects_per_label.get(k, 0) + objs_per_label[k]
            correct_per_label[k] = correct_per_label.get(k, 0) + crt_per_label[k]
        if obj_ok != len(gts_yolo):
            image_wrong_path.append(f"{pred_path.stem}.jpg")
        
    summary = {
        "image_total": image_total,
        "image_correct": image_correct,
        "image_acc": image_correct / max(1, image_total),
        "object_total": object_total,
        "object_correct": object_correct,
        "object_acc": (object_correct / object_total) if object_total > 0 else None,
        "objects_per_label": objects_per_label,
        "correct_per_label": correct_per_label,
        "acc_per_label": {k: (correct_per_label[k] / objects_per_label[k]) if objects_per_label[k] > 0 else None for k in objects_per_label},
        "image_wrong_path": image_wrong_path,
    }
    return summary


def run(input_folder):
    original_image_folder = Path(input_folder) / 'images'
    original_pred_folder = Path(input_folder) / 'pred_labels'
    original_pred_folder.mkdir(parents=True, exist_ok=True)

    output_folder = Path(input_folder) / 'results'
    img_paths = sorted([p for p in original_image_folder.iterdir() if p.suffix.lower() == ".jpg"])
    if not img_paths:
        print(f"未在 {input_folder} 中找到 .jpg 图像")
        return
    
    # # 1. 预测原始图像
    # batch_inference(img_paths, original_pred_folder)
    
    # # 2. 生成图像
    # for idx, img_path in enumerate(img_paths):
    #     image = cv2.imread(str(img_path))
    #     if image is None:
    #         print(f"无法读取图像: {img_path}")
    #         continue

    #     transforms = list(TRANSFORM_FUNCTION.keys())
    #     for transform in transforms:
    #         transform_function = TRANSFORM_FUNCTION[transform]
    #         transform_factors = TRANSFORM_CONFIG[transform]
    #         for factor in transform_factors:
    #             output_image_folder = output_folder / f"{transform}_{factor}" / 'images'
    #             if idx == 1:
    #                 output_image_folder.mkdir(parents=True, exist_ok=True)
    #             if 'fgsm' in transform:
    #                 if (Path(input_folder) / 'labels').exists():
    #                     reference_folder = Path(input_folder) / 'labels'
    #                 else:
    #                     reference_folder = Path(input_folder) / 'pred_labels'
                        
    #                 transformed_img = transform_function(image, str(reference_folder / f"{img_path.stem}.txt"), factor)
    #             else:
    #                 transformed_img = transform_function(image, factor)
    #             out_path = output_image_folder / img_path.name
    #             cv2.imwrite(str(out_path), transformed_img)
    
    # # # 3. 对所有生成的图像进行目标检测
    # transform_folders = sorted([p for p in output_folder.iterdir() if p.is_dir()])
    # for transform_folder in transform_folders:
    #     transform_img_folder = transform_folder / 'images'
    #     transform_img_paths = sorted([p for p in transform_img_folder.iterdir() if p.suffix.lower() == ".jpg"])
    #     if not transform_img_paths:
    #         print(f"未在 {transform_img_folder} 中找到 .jpg 图像")
    #         return
    #     transform_pred_folder = transform_folder / 'pred_labels'
    #     transform_pred_folder.mkdir(parents=True, exist_ok=True)
    #     batch_inference(transform_img_paths, transform_pred_folder)
    
    # 4. 将生成图像的识别结果与ground-truth（labels）或原始图像的检测结果（pred_labels）进行比较
    # TODO: False Positive need to be considered
    result_dict = {}
    if (Path(input_folder) / 'labels').exists():
        reference_folder = Path(input_folder) / 'labels'
    else:
        reference_folder = Path(input_folder) / 'pred_labels'
    
    is_xywh = False # pred_labels 是 xywh 格式或者是 xyxy 格式
    result_dict['original'] = acc_cal(original_pred_folder, reference_folder, original_image_folder, is_xywh)
    transform_folders = sorted([p for p in output_folder.iterdir() if p.is_dir()])
    for transform_folder in transform_folders:
        transform_img_folder = transform_folder / 'images'
        transform_label_folder = transform_folder / 'pred_labels'
        result_dict[transform_folder.name] = acc_cal(transform_label_folder, reference_folder, transform_img_folder, is_xywh)
    print(result_dict)
    with open(str(output_folder / 'summary.json'), "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

def extract_results_to_excel(project_path):
    # === 1. 读取文件 ===
    json_path = Path(project_path) / 'results' / 'summary.json'
    yaml_path = Path(project_path) / 'data.yaml'
    with open(json_path, "r") as f:
        data = json.load(f)
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    names = yaml_data.get("names", {})  # 可能是 list 或 dict
    
    # === 2. 构建结果列表 ===
    records = []
    for perturb_name, stats in data.items():
        # 每一行代表一种扰动方案
        row = {
            "perturbation": perturb_name,
            "image_acc": stats.get("image_acc", None),
            "object_acc": stats.get("object_acc", None)
        }

        # 读取 acc_per_label（每个类别的准确率）
        acc_per_label = stats.get("acc_per_label", {})
        for label, acc in acc_per_label.items():
            class_name = names[int(label)] 
            row[f"{class_name}"] = acc
        records.append(row)

    # === 3. 转换为 DataFrame ===
    df = pd.DataFrame(records)

    # === 4. 按列顺序排列（可选）===
    cols = ["perturbation", "image_acc", "object_acc"] + names
    df = df[cols]

    # === 5. 保存为 Excel 文件 ===
    output_path = Path(project_path) / 'results' / "accuracy_summary.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ 提取完成，共 {len(df)} 条扰动方案结果。")
    print(f"文件已保存为：{output_path}")
           
if __name__ == "__main__":
    # run("./demo_images")
    # run("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/dataset/aidx_upload_week4")
    extract_results_to_excel("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/dataset/aidx_upload_week4")



