import ultralytics
import torch
import cv2
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from ultralytics.utils.loss import v8DetectionLoss
import numpy as np
import os
import shutil

def apply_contrast(image, contrast_factor):
    """
    调整图像对比度
    """
    image = image.astype(np.float32) / 255.0
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * contrast_factor + mean
    image = np.clip(image, 0, 1.0)
    return (image * 255).astype(np.uint8)

# # 设置参数
# contrast_factor = 0.8
# input_folder = "./test_images/test/images"
# output_folder = f"./test_images/contrast_{contrast_factor}/images"
# # input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
# # label_folder = "./Vietnam-Traffic-Sign-Detection-6/test/labels"
# # output_folder = f"./Vietnam-Traffic-Sign-Detection-6/contrast_{contrast_factor}/images"
# os.makedirs(output_folder, exist_ok=True)

# # 对图像进行批处理
# for filename in os.listdir(input_folder):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)
#         image = cv2.imread(input_path)
#         if image is None:
#             print(f"无法读取图像：{input_path}")
#             continue
#         contrast_image = apply_contrast(image, contrast_factor)
#         cv2.imwrite(output_path, contrast_image)

# # 拷贝标签（保持原YOLO格式标签）
# shutil.copytree(label_folder, f"./Vietnam-Traffic-Sign-Detection-6/contrast_{contrast_factor}/labels")
