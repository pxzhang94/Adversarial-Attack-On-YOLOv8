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

# Attack: Rotation
def rotate_image(image, angle):
    """
    对图像进行中心旋转
    :param image: 输入图像（uint8）
    :param angle: 旋转角度（正值为逆时针）
    :return: 旋转后的图像
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

# # 设置参数
# rotation_angle = 5  # 旋转角度，正值为逆时针
# input_folder = "./test_images/test/images"
# output_folder = f"./test_images/rotated_{rotation_angle}/images"
# # input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
# # label_folder = "./Vietnam-Traffic-Sign-Detection-6/test/labels"
# # output_folder = f"./Vietnam-Traffic-Sign-Detection-6/rotated_{rotation_angle}/images"
# # label_output_folder = f"./Vietnam-Traffic-Sign-Detection-6/rotated_{rotation_angle}/labels"
# os.makedirs(output_folder, exist_ok=True)
# # os.makedirs(label_output_folder, exist_ok=True)

# # 对图像进行批处理
# for filename in os.listdir(input_folder):
#     if filename.lower().endswith(".jpg"):
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)
#         image = cv2.imread(input_path)
#         if image is None:
#             print(f"无法读取图像：{input_path}")
#             continue
#         rotated_image = rotate_image(image, angle=rotation_angle)
#         cv2.imwrite(output_path, rotated_image)

# for filename in os.listdir(label_folder):
#     if filename.lower().endswith(".txt"):
#         input_path = os.path.join(label_folder, filename)
#         output_path = os.path.join(label_output_folder, filename)
#         img_path = os.path.join(input_folder, filename[:-4]+'.jpg')
#         image = cv2.imread(img_path)
#         h, w = image.shape[:2]
#         rotated_bboxes = rotate_yolo_bboxes(input_path, rotation_angle, img_width=w, img_height=h)
#         save_yolo_bboxes_to_txt(rotated_bboxes, output_path)

