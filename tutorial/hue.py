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

def apply_hue(image, hue_angle):
    """
    调整图像色调（Hue）
    :param hue: 比例，例如 0.1 表示 ±18度（OpenCV的Hue范围是0-179）
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    image_hsv[..., 0] = (image_hsv[..., 0] + hue_angle) % 180
    image_out = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image_out


# # 设置参数
# hue_angle = 10  # from -179 to 179
# input_folder = "./test_images/test/images"
# label_folder = "./test_images/test/labels"
# output_folder = f"./test_images/hue_{hue_angle}/images"
# # input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
# # label_folder = "./Vietnam-Traffic-Sign-Detection-6/test/labels"
# # output_folder = f"./Vietnam-Traffic-Sign-Detection-6/hue_{hue_angle}/images"
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
#         hue_image = apply_hue(image, hue_angle)
#         cv2.imwrite(output_path, hue_image)

# # 拷贝标签（保持原YOLO格式标签）
# shutil.copytree(label_folder, f"./Vietnam-Traffic-Sign-Detection-6/hue_{hue_angle}/labels")
