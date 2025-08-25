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
import random

# TODO: num_blocks and block_size is defined by the size of image, maybe the input is a percentage
# Attack: Random Occlusion (with pixel-based block size)
def apply_random_occlusion(image, num_blocks=3, block_size=10):
    """
    在图像上添加随机遮挡块（单位：像素）
    :param image: 输入图像（uint8）
    :param num_blocks: 遮挡块数量
    :param block_size: 遮挡块最大宽或高（像素）
    :return: 遮挡后的图像
    """
    h, w = image.shape[:2]
    occluded = image.copy()

    for _ in range(num_blocks):
        block_w = block_size
        block_h = block_size
        if block_w >= w or block_h >= h:
            continue  # 跳过不合法的遮挡块尺寸
        x1 = random.randint(0, w - block_w)
        y1 = random.randint(0, h - block_h)
        # occluded[y1:y1 + block_h, x1:x1 + block_w] = np.random.randint(0, 256, (block_h, block_w, 3), dtype=np.uint8)
        occluded[y1:y1 + block_h, x1:x1 + block_w] = 0
    return occluded

# # 设置参数
# num_blocks = 5          # 遮挡块数量
# block_size = 20    # 每个遮挡块最大边长（单位：像素）
# input_folder = "./test_images/test/images"
# output_folder = f"./test_images/occlusion_no{num_blocks}_px{block_size}/images"
# # input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
# # label_folder = "./Vietnam-Traffic-Sign-Detection-6/test/labels"
# # output_folder = f"./Vietnam-Traffic-Sign-Detection-6/occlusion_no{num_blocks}_px{block_size}/images"
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
#         occluded_image = apply_random_occlusion(image, num_blocks=num_blocks, block_size=block_size)
#         cv2.imwrite(output_path, occluded_image)

# # 拷贝标签（保持原YOLO格式标签）
# shutil.copytree(label_folder, f"./Vietnam-Traffic-Sign-Detection-6/occlusion_no{num_blocks}_px{block_size}/labels")
