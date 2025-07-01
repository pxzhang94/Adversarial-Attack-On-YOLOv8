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

# Attack
def add_gaussian_noise(image, mean=0.0, std=0.1):
  image = image.astype(np.float32) / 255.0
  gaussian = np.random.normal(mean, std, image.shape).astype(np.float32)
  noisy_image = image + gaussian
  noisy_image = np.clip(noisy_image, 0.0, 1.0)  # 保持在0-1范围
  noisy_image = (noisy_image * 255).astype(np.uint8)  # 转换回uint8
  return noisy_image

std = 0.1
input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
output_folder = f"../adv_images/gaussian_images_{std}"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"无法读取图像：{input_path}")
            continue
        noisy_image = add_gaussian_noise(image, std=std)
        cv2.imwrite(output_path, noisy_image)