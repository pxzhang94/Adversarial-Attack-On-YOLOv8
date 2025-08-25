import sys
sys.path.append("../")
import os
import cv2
import math
import pandas as pd
from pathlib import Path
import yaml
import json

from utils.dataUtil import read_yolo_labels, yolo_to_xyxy, rotate_yolo_bboxes

# 可视化样式
COLOR_GT = (0, 200, 0)     # 绿色: GT
COLOR_PRED = (0, 128, 255) # 橙蓝: Pred
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
TEXT_THICK = 1

def draw_detections(image_bgr, gt_boxes, pred_boxes, gt_color=(0, 255, 0), pred_color=(0, 128, 255), class_names = None, title = None):
    """
    在 image_bgr 上叠加 YOLO 框，返回绘制后的副本。
    class_names 可选：用于把 class_id 显示成类别名；否则显示数字ID。
    """
    img = image_bgr.copy()
    H, W = img.shape[:2]
    for (cls, x, y, w, h) in gt_boxes:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy((cls, x, y, w, h), W, H)
        cv2.rectangle(img, (x1, y1), (x2, y2), gt_color, THICKNESS)
        label = class_names[cls_id] if (class_names and 0 <= cls_id < len(class_names)) else f"id:{cls_id}"
        # 文字背景
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICK)
        y_text = max(0, y1 - 5)
        cv2.rectangle(img, (x1, y_text - th - baseline), (x1 + tw, y_text + baseline), gt_color, -1)
        cv2.putText(img, label, (x1, y_text), FONT, FONT_SCALE, (255, 255, 255), TEXT_THICK, cv2.LINE_AA)
    
    for (cls, x, y, w, h) in pred_boxes:
        cls_id, x1, y1, x2, y2 = yolo_to_xyxy((cls, x, y, w, h), W, H)
        cv2.rectangle(img, (x1, y1), (x2, y2), pred_color, THICKNESS)
        label = class_names[cls_id] if (class_names and 0 <= cls_id < len(class_names)) else f"id:{cls_id}"
        # 文字背景
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICK)
        y_text = max(0, y1 - 5)
        cv2.rectangle(img, (x1, y_text - th - baseline), (x1 + tw, y_text + baseline), pred_color, -1)
        cv2.putText(img, label, (x1, y_text), FONT, FONT_SCALE, (255, 255, 255), TEXT_THICK, cv2.LINE_AA)

    if title:
        # 顶部左上角放标题条
        (tw, th), baseline = cv2.getTextSize(title, FONT, FONT_SCALE, TEXT_THICK)
        cv2.rectangle(img, (0, 0), (tw + 12, th + 10), (0, 0, 0), -1)
        cv2.putText(img, title, (6, th + 5), FONT, FONT_SCALE, (255, 255, 255), TEXT_THICK, cv2.LINE_AA)
    return img

def run(input_folder):
    with open(str(Path(input_folder) / "data.yaml"), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        class_names = data["names"]
        
    folder = Path(input_folder) / 'results'
    if (Path(input_folder) / 'labels').exists():
        reference_folder = Path(input_folder) / 'labels'
    else:
        reference_folder = Path(input_folder) / 'pred_labels'
    original_image_folder = Path(input_folder) / 'images'
    
    json_folder = folder / 'summary.json'
    with open(str(json_folder), "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    for condition, results in summary.items():
        img_list = results['image_wrong_path']
        
        if 'original' in condition:
            continue
        
        wrong_folder = folder / condition / 'image_wrong_predition'
        wrong_folder.mkdir(parents=True, exist_ok=True)
        for img_path in img_list:
            wrong_image_folder = wrong_folder / Path(img_path).stem
            wrong_image_folder.mkdir(parents=True, exist_ok=True)
            
            # 加载original image
            original_image_path = original_image_folder / img_path
            original_img = cv2.imread(str(original_image_path))
            print(img_path)
            
            gt_boxes = read_yolo_labels(reference_folder / (Path(img_path).stem + ".txt")) # read gt results
            original_boxes = read_yolo_labels(Path(input_folder) / 'pred_labels' / (Path(img_path).stem + ".txt")) # read gt results
            original_vis = draw_detections(original_img, gt_boxes, original_boxes, gt_color=COLOR_GT, pred_color=COLOR_PRED, class_names=class_names, title="Original Prediction")
            
            adv_image_path = folder / condition / 'images' / img_path
            adv_image = cv2.imread(str(adv_image_path))
            H, W = adv_image.shape[:2]
            if 'rotation' in condition:
                angle = int(condition.split('_')[-1])
                gt_boxes = rotate_yolo_bboxes(gt_boxes, angle, W, H)
            adv_boxes = read_yolo_labels(folder / condition / 'pred_labels' / (Path(img_path).stem + ".txt"))
            adv_vis = draw_detections(adv_image, gt_boxes, adv_boxes, gt_color=COLOR_GT, pred_color=COLOR_PRED, class_names=class_names, title="Perturbed Prediction")

            # 保存图片（保持原始文件名）
            out_original_path = wrong_image_folder / 'original.jpg'
            out_adv_path = wrong_image_folder / 'adversarial.jpg'
            cv2.imwrite(str(out_original_path), original_vis)
            cv2.imwrite(str(out_adv_path), adv_vis)
            
if __name__ == "__main__":
    run("./demo_images")
