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

# Attack: JPEG Compression
def apply_jpeg_compression(image, quality=30):
    """
    对图像进行JPEG压缩模拟伪影
    :param image: 输入图像（uint8）
    :param quality: 压缩质量（1-100，越低越模糊）
    :return: 模拟压缩后的图像
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg

# # 设置参数
# quality = 80  # JPEG压缩质量，值越小压缩越严重
# input_folder = "./test_images/test/images"
# output_folder = f"./test_images/jpeg_compression_q{quality}/images"
# # input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
# # label_folder = "./Vietnam-Traffic-Sign-Detection-6/test/labels"
# # output_folder = f"./Vietnam-Traffic-Sign-Detection-6/jpeg_compression_q{quality}/images"
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
#         compressed_image = apply_jpeg_compression(image, quality=quality)
#         cv2.imwrite(output_path, compressed_image)

# # 拷贝标签（保持原YOLO格式标签）
# shutil.copytree(label_folder, f"./Vietnam-Traffic-Sign-Detection-6/jpeg_compression_q{quality}/labels")
