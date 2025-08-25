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

from utils.config import TRANSFORM_CONFIG, TRANSFORM_FUNCTION
from utils.dataUtil import read_yolo_labels, write_yolo_labels, rotate_yolo_bboxes
from utils.matchUtil import match_and_score_single 






model = YOLO("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/tutorial/runs/detect/train/weights/best.pt")
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

def acc_cal(pred_folder, reference_folder, img_folder):
    image_total = 0
    image_correct = 0
    object_total = 0
    object_correct = 0
    image_wrong_path = []
    pred_paths = sorted([p for p in pred_folder.iterdir() if p.suffix.lower() == ".txt"])       
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
        img_ok, obj_tot, obj_ok = match_and_score_single(gts_yolo, preds_yolo, W, H, iou_thresh)
        image_total += 1
        image_correct += int(img_ok)
        object_total += obj_tot
        object_correct += obj_ok
        if obj_ok != len(gts_yolo):
            image_wrong_path.append(f"{pred_path.stem}.jpg")
        
    summary = {
        "image_total": image_total,
        "image_correct": image_correct,
        "image_acc": image_correct / max(1, image_total),
        "object_total": object_total,
        "object_correct": object_correct,
        "object_acc": (object_correct / object_total) if object_total > 0 else None,
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
    
    # 预测原始图像
    batch_inference(img_paths, original_pred_folder)
    
    # 生成图像
    for idx, img_path in enumerate(img_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue

        transforms = list(TRANSFORM_FUNCTION.keys())
        for transform in transforms:
            transform_function = TRANSFORM_FUNCTION[transform]
            transform_factors = TRANSFORM_CONFIG[transform]
            for factor in transform_factors:
                output_image_folder = output_folder / f"{transform}_{factor}" / 'images'
                if idx == 1:
                    output_image_folder.mkdir(parents=True, exist_ok=True)
                if 'fgsm' in transform:
                    if (Path(input_folder) / 'labels').exists():
                        reference_folder = Path(input_folder) / 'labels'
                    else:
                        reference_folder = Path(input_folder) / 'pred_labels'
                        
                    transformed_img = transform_function(image, str(reference_folder / f"{img_path.stem}.txt"), factor)
                else:
                    transformed_img = transform_function(image, factor)
                out_path = output_image_folder / img_path.name
                cv2.imwrite(str(out_path), transformed_img)
    
    # # 对所有生成的图像进行目标检测
    transform_folders = sorted([p for p in output_folder.iterdir() if p.is_dir()])
    for transform_folder in transform_folders:
        transform_img_folder = transform_folder / 'images'
        transform_img_paths = sorted([p for p in transform_img_folder.iterdir() if p.suffix.lower() == ".jpg"])
        if not transform_img_paths:
            print(f"未在 {transform_img_folder} 中找到 .jpg 图像")
            return
        transform_pred_folder = transform_folder / 'pred_labels'
        transform_pred_folder.mkdir(parents=True, exist_ok=True)
        batch_inference(transform_img_paths, transform_pred_folder)
    
    # 将生成图像的识别结果与ground-truth（labels）或原始图像的检测结果（pred_labels）进行比较
    # TODO: False Positive need to be considered
    result_dict = {}
    if (Path(input_folder) / 'labels').exists():
        reference_folder = Path(input_folder) / 'labels'
    else:
        reference_folder = Path(input_folder) / 'pred_labels'
        
    result_dict['original'] = acc_cal(original_pred_folder, reference_folder, original_image_folder)
    for transform_folder in transform_folders:
        transform_img_folder = transform_folder / 'images'
        transform_label_folder = transform_folder / 'pred_labels'
        result_dict[transform_folder.name] = acc_cal(transform_label_folder, reference_folder, transform_img_folder)
    print(result_dict)
    with open(str(output_folder / 'summary.json'), "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
            
if __name__ == "__main__":
    run("./demo_images")