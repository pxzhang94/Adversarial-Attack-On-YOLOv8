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

# Attack: Brightness Adjustment
def apply_brightness(image, brightness_factor=1.2):
    """
    调整图像亮度
    :param image: 输入图像（uint8）
    :param brightness_factor: 亮度因子，>1 增亮，<1 变暗
    :return: 调整后的图像
    """
    image = image.astype(np.float32)
    adjusted = image * brightness_factor
    adjusted = np.clip(adjusted, 0, 255)
    return adjusted.astype(np.uint8)

# def run():
#     # 设置参数
#     brightness_factor = 1.2  # >1 增亮，<1 变暗（例如 0.6）
#     input_folder = "./test_images/test/images"
#     output_folder = f"./test_images/brightness_{brightness_factor}/images"
#     # input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
#     # label_folder = "./Vietnam-Traffic-Sign-Detection-6/test/labels"
#     # output_folder = f"./Vietnam-Traffic-Sign-Detection-6/brightness_{brightness_factor}/images"
#     os.makedirs(output_folder, exist_ok=True)

#     # 对图像进行批处理
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(".jpg"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)
#             image = cv2.imread(input_path)
#             if image is None:
#                 print(f"无法读取图像：{input_path}")
#                 continue
#             bright_image = apply_brightness(image, brightness_factor=brightness_factor)
#             cv2.imwrite(output_path, bright_image)

#     # 拷贝标签（保持原YOLO格式标签）
#     shutil.copytree(label_folder, f"./Vietnam-Traffic-Sign-Detection-6/brightness_{brightness_factor}/labels")

# if __name__ == "__main__":
#     run()