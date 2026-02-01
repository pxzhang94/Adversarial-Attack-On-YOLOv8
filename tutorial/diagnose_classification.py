import sys
sys.path.append("../")
sys.path.append("/root/autodl-tmp/project/") # 修改为你的项目(Model)根目录路径
import os
import cv2
import json
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import yaml
import csv

from utils.config_classification import TRANSFORM_CONFIG, TRANSFORM_FUNCTION
from utils.dataUtil import normalize_data

def batch_inference(model, img_paths, input_size, output_label_path, normalize=None, mean=None, std=None, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    mean, std = normalize_data(device, normalize, mean, std)
    rows = [("filename", "label")]
    model.eval()
    
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

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            rows.append((img_path.name, pred.detach().cpu().item()))
    with open(output_label_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def acc_cal(reference_file, ori_pred_file, pred_file):
    ref_df  = pd.read_csv(reference_file)
    ori_pred_df = pd.read_csv(ori_pred_file)
    pred_df = pd.read_csv(pred_file)
    
    ref_df["filename"]  = ref_df["filename"].astype(str).str.strip().apply(lambda x: Path(x).stem)
    ori_pred_df["filename"] = ori_pred_df["filename"].astype(str).str.strip().apply(lambda x: Path(x).stem)
    pred_df["filename"] = pred_df["filename"].astype(str).str.strip().apply(lambda x: Path(x).stem)
    
    ref_df = ref_df.rename(columns={"label": "label_gt"})
    ori_pred_df = ori_pred_df.rename(columns={"label": "label_ori_pred"})
    pred_df = pred_df.rename(columns={"label": "label_pred"})
    
    merged_df = (ref_df.merge(ori_pred_df, on="filename", how="left").merge(pred_df, on="filename", how="left"))
    
    image_total = int(len(merged_df))
    image_correct = int((merged_df["label_gt"] == merged_df["label_pred"]).sum())
    df_ori_correct = merged_df[merged_df["label_ori_pred"] == merged_df["label_gt"]]
    num_ori_correct = len(df_ori_correct)
    df_flip = df_ori_correct[df_ori_correct["label_ori_pred"] != df_ori_correct["label_pred"]]
    diff_filenames = df_flip["filename"].dropna().tolist()
    num_flip = len(df_flip)
        
    summary = {
        "image_total": image_total,
        "image_correct": image_correct,
        "image_acc": image_correct / max(1, image_total),
        "robustness": round(1 - num_flip / max(1, num_ori_correct), 4),
        "image_diff_file": diff_filenames,
    }
    return summary


def run(model_file=None, input_folder=None, input_size=None, normalize=None, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], mode='upload', model_desc_file_no=None, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    original_image_folder = Path(input_folder) / 'images'
    original_pred_path = Path(input_folder) / 'pred_labels.csv'

    output_folder = Path(input_folder) / 'results'
    img_paths = sorted([p for p in original_image_folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not img_paths:
        print(f"Do not find image in {input_folder}")
        return
    
    if 'upload' == mode:
        import importlib
        module_name = 'model_' + str(model_desc_file_no)
        module = importlib.import_module(f"model_desc.{module_name}")
        
        model_desc = getattr(module, 'model')
        model = model_desc().to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
    elif 'api' == mode:
        #TODO: load your model here
        return None
    
    # 1. 预测原始图像
    batch_inference(model, img_paths, input_size, original_pred_path, normalize, mean, std, device)
    
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
                if transform in ['fgsm', 'pgd', 'cw']:
                    if (Path(input_folder) / 'labels.csv').exists():
                        df = pd.read_csv(str(Path(input_folder) / 'labels.csv'))
                    else:
                        df = pd.read_csv(str(Path(input_folder) / 'pred_labels.csv'))
                    img2label = dict(zip(df["filename"], df["label"]))
                    label =img2label[img_path.name]
                    
                    transformed_img = transform_function(model, image, label, factor, normalize, mean, std, device)
                else:
                    transformed_img = transform_function(image, factor)
                
                stored_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_image_folder / img_path.stem)+'.png', stored_img)
    
    # 3. 对所有生成的图像进行目标检测
    transform_folders = sorted([p for p in output_folder.iterdir() if p.is_dir()])
    for transform_folder in transform_folders:
        transform_img_folder = transform_folder / 'images'
        transform_img_paths = sorted([p for p in transform_img_folder.iterdir() if p.suffix.lower() == ".png"])
        if not transform_img_paths:
            print(f"Can not find .png image in {transform_img_folder}")
            return
        transform_pred_path = transform_folder / 'pred_labels.csv'
        batch_inference(model, transform_img_paths, input_size, transform_pred_path, normalize, mean, std, device)
    
    # 4. 将生成图像的识别结果与ground-truth（labels）或原始图像的检测结果（pred_labels）进行比较
    result_dict = {}
    if (Path(input_folder) / 'labels.csv').exists():
        reference_file = Path(input_folder) / 'labels.csv'
    else:
        reference_file = Path(input_folder) / 'pred_labels.csv'
    
    result_dict['original'] = acc_cal(reference_file, original_pred_path, original_pred_path)
    transform_folders = sorted([p for p in output_folder.iterdir() if p.is_dir()])
    for transform_folder in transform_folders:
        transform_label_file = transform_folder / 'pred_labels.csv'
        result_dict[transform_folder.name] = acc_cal(reference_file, original_pred_path, transform_label_file)
        
    with open(str(output_folder / 'diagnose_results.json'), "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

def extract_results_to_excel(project_path):
    # === 1. 读取文件 ===
    json_path = Path(project_path) / 'results' / 'diagnose_results.json'
    # yaml_path = Path(project_path) / 'data.yaml'
    with open(json_path, "r") as f:
        data = json.load(f)
    # with open(yaml_path, "r") as f:
    #     yaml_data = yaml.safe_load(f)
    # names = yaml_data.get("names", {})  # 可能是 list 或 dict
    
    # === 2. 构建结果列表 ===
    records = []
    for perturb_name, stats in data.items():
        # 每一行代表一种扰动方案
        row = {
            "perturbation": perturb_name,
            "image_acc": stats.get("image_acc", None),
        }

        # # 读取 acc_per_label（每个类别的准确率）
        # acc_per_label = stats.get("acc_per_label", {})
        # for label, acc in acc_per_label.items():
        #     class_name = names[int(label)] 
        #     row[f"{class_name}"] = acc
        records.append(row)

    # === 3. 转换为 DataFrame ===
    df = pd.DataFrame(records)

    # === 4. 按列顺序排列（可选）===
    cols = ["perturbation", "image_acc"]
    # cols = ["perturbation", "image_acc"] + names
    df = df[cols]

    # === 5. 保存为 Excel 文件 ===
    output_path = Path(project_path) / 'results' / "diagnose_results.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ 提取完成，共 {len(df)} 条扰动方案结果。")
    print(f"文件已保存为：{output_path}")
           
if __name__ == "__main__":
    run(model_file='./vgg_cifar10_best.pt', input_folder="./demo_classification", normalize='CIFAR10', model_desc_file_no=1768547474, input_size=(32,32), device = 'cuda' if torch.cuda.is_available() else 'cpu')
    # run("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/dataset/aidx_upload_week4")
    # extract_results_to_excel("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/tutorial/demo_classification")



