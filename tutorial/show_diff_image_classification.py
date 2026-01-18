import sys
sys.path.append("../")
import os
import cv2
import math
import pandas as pd
from pathlib import Path
import yaml
import json
import shutil

from utils.dataUtil import read_yolo_labels, yolo_to_xyxy, rotate_yolo_bboxes

# 可视化样式
COLOR_GT = (0, 200, 0)     # 绿色: GT
COLOR_PRED = (0, 128, 255) # 橙蓝: Pred
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
TEXT_THICK = 1

def draw_label(image_bgr, title = None):
    img = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)

    if title:
        # 顶部左上角放标题条
        (tw, th), baseline = cv2.getTextSize(title, FONT, FONT_SCALE, TEXT_THICK)
        cv2.rectangle(img, (0, 0), (tw + 12, th + 10), (0, 0, 0), -1)
        cv2.putText(img, title, (6, th + 5), FONT, FONT_SCALE, (255, 255, 255), TEXT_THICK, cv2.LINE_AA)
    return img

def run(input_folder):
    suffixes = [".png", ".jpg", ".jpeg"]
    
    with open(str(Path(input_folder) / "data.yaml"), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        class_names = data["names"]
        
    folder = Path(input_folder) / 'results'
    if (Path(input_folder) / 'labels.csv').exists():
        reference_file = Path(input_folder) / 'labels.csv'
    else:
        reference_file = Path(input_folder) / 'pred_labels.csv'
    original_label_df = pd.read_csv(str(reference_file))
    original_image_folder = Path(input_folder) / 'images'
    
    json_folder = folder / 'diagnose_results.json'
    with open(str(json_folder), "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    for condition, results in summary.items():
        img_list = results['image_diff_file']
        
        if 'original' in condition:
            continue
        
        diff_folder = folder / condition / 'image_diff_predition'
        if diff_folder.exists() and diff_folder.is_dir():
            shutil.rmtree(diff_folder)
        diff_folder.mkdir(parents=True, exist_ok=True)
        for img_file in img_list:
            diff_image_folder = diff_folder / Path(img_file)
            diff_image_folder.mkdir(parents=True, exist_ok=True)
            
            # 加载original image
            for suf in suffixes:
                original_image_path = original_image_folder / (img_file + suf)
                if original_image_path.exists():
                    break
            original_img = cv2.imread(str(original_image_path))
            original_label = int(original_label_df.loc[original_label_df["filename"] == original_image_path.name, "label"].iloc[0])
            original_vis = draw_label(original_img, title="Original Image:"+ class_names[original_label])
            
            adv_image_path = folder / condition / 'images' / (img_file+'.png')
            adv_image = cv2.imread(str(adv_image_path))
            adv_label_path = folder / condition / 'pred_labels.csv'
            adv_label_df = pd.read_csv(str(adv_label_path))
            adv_label = int(adv_label_df.loc[adv_label_df["filename"] == (img_file+'.png'), "label"].iloc[0])
            adv_vis = draw_label(adv_image, title="Adversarial Image:"+ class_names[adv_label])
            
            # 保存图片（保持原始文件名）
            out_original_path = diff_image_folder / 'original.jpg'
            out_adv_path = diff_image_folder / 'adversarial.jpg'
            cv2.imwrite(str(out_original_path), original_vis)
            cv2.imwrite(str(out_adv_path), adv_vis)
            
if __name__ == "__main__":
    run("./demo_classification")
    # run("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/dataset/aidx_round_2_test_dataset")
    # run("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/dataset/aidx_upload_week4")
