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

# Attack: Gaussian Blur
def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    对图像应用高斯模糊
    :param image: 输入图像（uint8）
    :param kernel_size: 高斯核大小（必须是奇数，例如3, 5, 7）
    :param sigma: 标准差
    :return: 模糊后的图像
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred

# # 设置参数
# kernel_size = 5
# sigma = 2.0
# input_folder = "./test_images/test/images"
# output_folder = f"./test_images/gaussian_blur_k{kernel_size}_s{sigma}/images"
# # input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
# # label_folder = "./Vietnam-Traffic-Sign-Detection-6/test/labels"
# # output_folder = f"./Vietnam-Traffic-Sign-Detection-6/gaussian_blur_k{kernel_size}_s{sigma}/images"
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
#         blurred_image = apply_gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
#         cv2.imwrite(output_path, blurred_image)
# shutil.copytree(label_folder, f"./Vietnam-Traffic-Sign-Detection-6/gaussian_blur_k{kernel_size}_s{sigma}/labels")