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

def apply_saturation(image, saturation_factor):
    """
    调整图像饱和度（通过HSV空间）
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    image_hsv[..., 1] *= saturation_factor
    image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)
    image_out = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image_out

# # 设置参数
# saturation_factor = 0.8
# input_folder = "./test_images/test/images"
# output_folder = f"./test_images/saturation_{saturation_factor}/images"
# # input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
# # label_folder = "./Vietnam-Traffic-Sign-Detection-6/test/labels"
# # output_folder = f"./Vietnam-Traffic-Sign-Detection-6/saturation_{saturation_factor}/images"
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
#         saturation_image = apply_saturation(image, saturation_factor)
#         cv2.imwrite(output_path, saturation_image)

# # 拷贝标签（保持原YOLO格式标签）
# shutil.copytree(label_folder, f"./Vietnam-Traffic-Sign-Detection-6/saturation_{saturation_factor}/labels")
