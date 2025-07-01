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
from utils.datsUtil import convert_yolo_to_batch_format_torch
import shutil

# Attack
def add_fgsm_noise(trainer, loss_fn, image, labels_dict, epsilon=0.1):
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

args = dict(model="/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/tutorial/runs/detect/train/weights/best.pt", data="./Vietnam-Traffic-Sign-Detection-6/data.yaml", epochs=100)
trainer = DetectionTrainer(overrides=args)
trainer.setup_model()
trainer.set_model_attributes()
trainer.model.eval()
loss_fn = v8DetectionLoss(trainer.model)

epsilon = 0.1
input_folder = "./Vietnam-Traffic-Sign-Detection-6/test/images"
label_folder = "./Vietnam-Traffic-Sign-Detection-6/test/labels"
output_folder = f"./Vietnam-Traffic-Sign-Detection-6/fgsm_{epsilon}/images"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        input_path = os.path.join(input_folder, filename)
        label_path = os.path.join(label_folder, filename.replace(".jpg", ".txt"))
        output_path = os.path.join(output_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"无法读取图像：{input_path}")
            continue
        anotation = convert_yolo_to_batch_format_torch(label_path)
        if anotation is None:
          continue
        batchidx, cls, bbox = anotation
        labels_dict = {
          'batch_idx': batchidx,  # Assuming first column indicates batch_idx
          'cls': cls,  # Assuming second column is class
          'bboxes': bbox  # Assuming last columns are bbox coordinates
        }
        noisy_image = add_fgsm_noise(trainer, loss_fn, image, labels_dict, epsilon)
        cv2.imwrite(output_path, noisy_image)
shutil.copytree(label_folder, f"./Vietnam-Traffic-Sign-Detection-6/fgsm_{epsilon}/labels")