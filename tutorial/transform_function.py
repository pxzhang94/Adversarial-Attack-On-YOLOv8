import sys
sys.path.append('../')
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
from utils.dataUtil import convert_yolo_to_batch_format_torch

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

def apply_contrast(image, contrast_factor):
    """
    调整图像对比度
    """
    image = image.astype(np.float32) / 255.0
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    image = (image - mean) * contrast_factor + mean
    image = np.clip(image, 0, 1.0)
    return (image * 255).astype(np.uint8)

# Attack: Gaussian Blur
def apply_gaussian_blur(image, sigma=1.0, kernel_size=5):
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

# Attack: Gaussian Noise
def apply_gaussian_noise(image, std=0.1, mean=0.0):
  image = image.astype(np.float32) / 255.0
  gaussian = np.random.normal(mean, std, image.shape).astype(np.float32)
  noisy_image = image + gaussian
  noisy_image = np.clip(noisy_image, 0.0, 1.0)  # 保持在0-1范围
  noisy_image = (noisy_image * 255).astype(np.uint8)  # 转换回uint8
  return noisy_image

def apply_hue(image, hue_angle):
    """
    调整图像色调（Hue）
    :param hue: 比例，例如 0.1 表示 ±18度（OpenCV的Hue范围是0-179）
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    image_hsv[..., 0] = (image_hsv[..., 0] + hue_angle) % 180
    image_out = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image_out

# Attack: Rotation
def apply_rotation(image, angle):
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

def apply_saturation(image, saturation_factor):
    """
    调整图像饱和度（通过HSV空间）
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    image_hsv[..., 1] *= saturation_factor
    image_hsv[..., 1] = np.clip(image_hsv[..., 1], 0, 255)
    image_out = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image_out

# Attack: FGSM
def apply_fgsm(image, label_path, epsilon=0.1):
    args = dict(model="/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/tutorial/runs/detect/train/weights/best.pt", data="./Vietnam-Traffic-Sign-Detection-6/data.yaml", epochs=100)
    trainer = DetectionTrainer(overrides=args)
    trainer.setup_model()
    trainer.set_model_attributes()
    trainer.model.eval()
    loss_fn = v8DetectionLoss(trainer.model)
    
    anotation = convert_yolo_to_batch_format_torch(label_path)
    if anotation is None:
        return image
    batchidx, cls, bbox = anotation
    labels_dict = {
        'batch_idx': batchidx,  # Assuming first column indicates batch_idx
        'cls': cls,  # Assuming second column is class
        'bboxes': bbox  # Assuming last columns are bbox coordinates
    }
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    
    image.requires_grad = True
    pred = trainer.model(image.unsqueeze(0))
    loss, _ = loss_fn(pred, labels_dict)
    loss = loss.mean()

    trainer.model.zero_grad()
    loss.backward()

    perturbed_image = image + epsilon * image.grad.data.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    perturbed_image = perturbed_image.squeeze(0).permute(1, 2, 0).detach().numpy() * 255.0
    perturbed_image = perturbed_image.astype(np.uint8)
    perturbed_image = cv2.cvtColor(perturbed_image, cv2.COLOR_RGB2BGR)
    return perturbed_image