import sys
sys.path.append("../")
sys.path.append("/root/autodl-tmp/project/") # 修改为你的项目(Model)根目录路径

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
from utils.matchUtil import match_and_score_single, decode_prediction
from utils.dataUtil import normalize_data

def batch_inference(model, img_paths, input_size, output_label_dir, normalize=None, mean=None, std=None, device='CUDA' if torch.cuda.is_available() else 'CPU', conf_thresh = 0.3, iou_thresh = 0.5):
    mean, std = normalize_data(device, normalize, mean, std)
    
    for i in range(len(img_paths)):
        img_path = img_paths[i]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Can not load image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        img = img.astype(np.float32) / 255.0

        x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        x = (x - mean) / std
        
        preds = model(x)
        
        # # 兼容某些模型返回 tuple/list/dict
        # if isinstance(preds, (tuple, list)):
        #     preds = preds[0]
        # elif isinstance(preds, dict):
        #     # 常见 key：'pred', 'boxes' 等；按需改
        #     preds = preds.get("pred", preds.get("boxes", None))
        #     if preds is None:
        #         raise ValueError("Model returned dict but no 'pred'/'boxes' key found.")
        preds = decode_prediction(preds, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            
        # 转成 numpy (N,5)
        if torch.is_tensor(preds):
            preds = preds.detach().cpu().numpy()
        else:
            preds = np.asarray(preds)

        if preds.ndim != 2 or preds.shape[1] != 6:
            raise ValueError(f"Expected preds shape (N,5), got {preds.shape}")

        label_txt = output_label_dir / f"{img_path.stem}.txt"
        with open(label_txt, "w", encoding="utf-8") as f:
            for row in preds:
                c = int(row[0])
                xc, yc, w, h = row[1] / input_size[1], row[2] / input_size[0], row[3] / input_size[1], row[4] / input_size[0]
                conf = row[5]
                f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

def acc_cal(reference_folder, ori_pred_folder, pred_folder, img_folder, iou_thresh=0.5, is_xywh=True):
    image_total = 0
    image_correct = 0
    object_total = 0
    object_correct = 0
    object_ori_correct = 0
    object_robust = 0
    image_diff_file = []
    pred_paths = sorted([p for p in pred_folder.iterdir() if p.suffix.lower() == ".txt"])   
    objects_per_label = {}
    correct_per_label = {}   
    ori_correct_per_label = {}
    robust_per_label = {} 
    for pred_path in pred_paths:
        img_path = img_folder / f"{pred_path.stem}.jpg"
        if img_path.exists() == False:
            img_path = img_folder / f"{pred_path.stem}.png"
        reference_path = reference_folder / pred_path.name
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Can not load image: {img_path}")
            continue
        H, W = img.shape[:2]  # 高, 宽
        
        gts_yolo = read_yolo_labels(reference_path)
        pred_path_str = str(pred_path.absolute())
        if 'rotation' in pred_path_str:
            angle = int(pred_path_str.split('/')[-3].split('_')[-1])
            gts_yolo = rotate_yolo_bboxes(gts_yolo, angle, W, H)
        
        ori_yolo = read_yolo_labels(ori_pred_folder / pred_path.name)
        
        preds_yolo = read_yolo_labels(pred_path)
        img_ok, obj_tot, obj_ok, obj_ori_ok, obj_robust, objs_per_label, crt_per_label, ori_ok_per_label, rob_per_label = match_and_score_single(gts_yolo, ori_yolo, preds_yolo, W, H, iou_thresh, is_xywh)
        # img_ok, obj_tot, obj_ok, objs_per_label, crt_per_label = match_and_score_single(gts_yolo, ori_yolo, preds_yolo, W, H, iou_thresh, is_xywh)
        image_total += 1
        image_correct += int(img_ok)
        object_total += obj_tot
        object_correct += obj_ok
        object_ori_correct += obj_ori_ok
        object_robust += obj_robust
        for k in objs_per_label.keys():
            objects_per_label[k] = objects_per_label.get(k, 0) + objs_per_label[k]
            correct_per_label[k] = correct_per_label.get(k, 0) + crt_per_label[k]
            ori_correct_per_label[k] = ori_correct_per_label.get(k, 0) + ori_ok_per_label[k]
            robust_per_label[k] = robust_per_label.get(k, 0) + rob_per_label[k]
        if obj_robust > 0:
            image_diff_file.append(f"{pred_path.stem}.jpg")
        
    summary = {
        "image_total": image_total,
        "image_correct": image_correct,
        "image_acc": image_correct / max(1, image_total),
        "object_total": object_total,
        "object_correct": object_correct,
        "object_acc": (round(object_correct / object_total, 4)) if object_total > 0 else None,
        "object_ori_correct": object_ori_correct,
        "object_robust": object_robust,
        "robustness": (round(1 - object_robust / max(1, object_ori_correct), 4)) if object_ori_correct > 0 else None,
        "objects_per_label": objects_per_label,
        "correct_per_label": correct_per_label,
        "ori_correct_per_label": ori_correct_per_label,
        "robust_per_label": robust_per_label,
        "acc_per_label": {k: (correct_per_label[k] / objects_per_label[k]) if objects_per_label[k] > 0 else None for k in objects_per_label},
        "robustness_per_label": {k: (round(1 - robust_per_label[k] / max(1, ori_correct_per_label[k]), 4)) if ori_correct_per_label[k] > 0 else None for k in objects_per_label},
        "image_diff_file": image_diff_file,
    }
    return summary


def run(model_file=None, input_folder=None, input_size=None, normalize=None, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], mode='upload', model_desc_file_no=None, device='CUDA' if torch.cuda.is_available() else 'CPU', conf_thresh = 0.3, iou_thresh = 0.5):
    original_image_folder = Path(input_folder) / 'images'
    original_pred_folder = Path(input_folder) / 'pred_labels'
    original_pred_folder.mkdir(parents=True, exist_ok=True)

    output_folder = Path(input_folder) / 'results_test'
    img_paths = sorted([p for p in original_image_folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not img_paths:
        print(f"Do not find image in {input_folder}")
        return
        
    if 'upload' == mode:
        #TODO: Check is it right?
        import importlib
        module_name = 'model_' + str(model_desc_file_no)
        module = importlib.import_module(f"model_desc.{module_name}")
        
        if hasattr(module, "model"):
            model_desc = getattr(module, "model")
            model = model_desc().to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
        elif hasattr(module, "adapter"):
            model_desc = getattr(module, "adapter")
            model = model_desc(model_file, device)
        else:
            raise AttributeError("Module must define either 'model' or 'adapter'")

        model.eval()
    elif 'api' == mode:
        #TODO: load your model here
        return None
    
    # 1. 预测原始图像
    batch_inference(model, img_paths, input_size, original_pred_folder, normalize, mean, std, device, conf_thresh, iou_thresh)
    
    # 2. 生成图像
    for idx, img_path in enumerate(img_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Can not load image: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, input_size)

        transforms = list(TRANSFORM_FUNCTION.keys())
        for transform in transforms:
            transform_function = TRANSFORM_FUNCTION[transform]
            transform_factors = TRANSFORM_CONFIG[transform]
            for factor in transform_factors:
                output_image_folder = output_folder / f"{transform}_{factor}" / 'images'
                if idx == 0:
                    output_image_folder.mkdir(parents=True, exist_ok=True)
                if transform in ['togm', 'togv', 'togf']:
                    if (Path(input_folder) / 'labels').exists():
                        reference_folder = Path(input_folder) / 'labels'
                    else:
                        reference_folder = Path(input_folder) / 'pred_labels'
                        
                    # transformed_img = transform_function(image, str(reference_folder / f"{img_path.stem}.txt"), factor)
                    transformed_img = transform_function(model, image, factor, normalize, mean, std, device)
                else:
                    transformed_img = transform_function(image, factor)
                
                stored_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_image_folder / img_path.stem)+'.png', stored_img)
    print(f"Images saved in {output_folder}")
    
    # 3. 对所有生成的图像进行目标检测
    transform_folders = sorted([p for p in output_folder.iterdir() if p.is_dir()])
    for transform_folder in transform_folders:
        transform_img_folder = transform_folder / 'images'
        transform_img_paths = sorted([p for p in transform_img_folder.iterdir() if p.suffix.lower() == ".png"])
        if not transform_img_paths:
            print(f"Can not find .png image in {transform_img_folder}")
            return
        transform_pred_folder = transform_folder / 'pred_labels'
        transform_pred_folder.mkdir(parents=True, exist_ok=True)
        batch_inference(model, transform_img_paths, input_size, transform_pred_folder, normalize, mean, std, device, conf_thresh, iou_thresh)

    # 4. 将生成图像的识别结果与ground-truth（labels）或原始图像的检测结果（pred_labels）进行比较
    # TODO: False Positive need to be considered
    result_dict = {}
    if (Path(input_folder) / 'labels').exists():
        reference_folder = Path(input_folder) / 'labels'
    else:
        reference_folder = Path(input_folder) / 'pred_labels'
    
    is_xywh = True # pred_labels 是 xywh 格式或者是 xyxy 格式
    result_dict['original'] = acc_cal(reference_folder, original_pred_folder, original_pred_folder, original_image_folder, iou_thresh, is_xywh)
    transform_folders = sorted([p for p in output_folder.iterdir() if p.is_dir()])
    for transform_folder in transform_folders:
        transform_img_folder = transform_folder / 'images'
        transform_label_folder = transform_folder / 'pred_labels'
        result_dict[transform_folder.name] = acc_cal(reference_folder, original_pred_folder, transform_label_folder, transform_img_folder, iou_thresh, is_xywh)
    print(result_dict)
    with open(str(output_folder / 'diagnose_results.json'), "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

def extract_results_to_excel(project_path):
    # === 1. 读取文件 ===
    json_path = Path(project_path) / 'results' / 'diagnose_results.json'
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
    print("begin generate attack images...")
    run(model_file='/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/model/yolov8m.pt', input_folder="./demo_images", model_desc_file_no=1768547475, input_size=(640,640), device = 'cuda' if torch.cuda.is_available() else 'cpu', conf_thresh=0.3, iou_thresh=0.5)
    print("Done!")
    # run("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/dataset/aidx_upload_week4")
    # extract_results_to_excel("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/dataset/aidx_upload_week4")



