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
args = dict(model="/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/tutorial/runs/detect/train/weights/best.pt", data="../datasets/data.yaml", epochs=100)
trainer = DetectionTrainer(overrides=args)
trainer.setup_model()
trainer.set_model_attributes()
trainer.model.eval()

#Load images and Load labels
def load_image(img_path):
  img_path = str(img_path)
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
  img = torch.tensor(img).permute(2,0,1) / 255.0  # Normalize and permute dimensions
  return img

def load_label(labels_path):
  labels = torch.tensor([list(map(float, line.split())) for line in open(labels_path)])
  return labels

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        super().__init__()
        self.image_paths = list(Path(image_dir).glob('*.jpg'))  # Assuming JPG images
        self.label_paths = [Path(label_dir)/f'{p.stem}.txt' for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = load_image(self.image_paths[idx])
        # img_path = str(self.image_paths[idx])
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        # img = torch.tensor(img).permute(2,0,1) / 255.0  # Normalize and permute dimensions

        # Load labels
        labels = load_label(self.label_paths[idx])
        # label_path = self.label_paths[idx]
        # labels = torch.tensor([list(map(float, line.split())) for line in open(label_path)])

        return img, labels

def convert_yolo_to_batch_format_torch(label_file):
    # Load YOLO format labels
    labels = np.loadtxt(label_file)
    if labels.size == 0:
          return None

    # Initialize lists for batch_idx, cls, and bboxes
    batch_indices = []
    classes = []
    bboxes = []

    if len(labels.shape) >= 2:
      for label in labels:
          #print(label)
          cls, x_center, y_center, width, height = label

          # Set batch index to 0 for all entries
          batch_indices.append(0)

          # Add class ID
          classes.append(int(cls))

          # Convert YOLO bbox to [x_min, y_min, x_max, y_max]
          x_min = x_center - width / 2.0
          y_min = y_center - height / 2.0
          x_max = x_center + width / 2.0
          y_max = y_center + height / 2.0

          # Add bbox
          bboxes.append([x_min, y_min, x_max, y_max])
    elif (len(labels.shape)==1):
          cls, x_center, y_center, width, height = labels

          # Set batch index to 0 for all entries
          batch_indices.append(0)

          # Add class ID
          classes.append(int(cls))

          # Convert YOLO bbox to [x_min, y_min, x_max, y_max]
          x_min = x_center - width / 2.0
          y_min = y_center - height / 2.0
          x_max = x_center + width / 2.0
          y_max = y_center + height / 2.0

          # Add bbox
          bboxes.append([x_min, y_min, x_max, y_max])
    else:
          return None

    # Convert to torch tensors
    batch_indices_tensor = torch.tensor(batch_indices).view(-1, 1)
    classes_tensor = torch.tensor(classes).view(-1, 1)
    bboxes_tensor = torch.tensor(bboxes).view(-1, 4)  # 4 because bboxes are [x_min, y_min, x_max, y_max]

    return batch_indices_tensor, classes_tensor, bboxes_tensor

def ComputeGradient(image_dir, label_dir):
  dataset = CustomDataset(image_dir, label_dir)
  loader = DataLoader(dataset, batch_size=1, shuffle=False)
  loss_fn = v8DetectionLoss(trainer.model)

  labelsfile = set()

  for root, dirs, files in os.walk(label_dir):
      for file in files:
        if file.endswith('.txt'):
          full_path = os.path.join(root, file)
          labelsfile.add(full_path)

  # Now, you can iterate over the loader
  for img_batch, labels_batch in loader:
      temp = convert_yolo_to_batch_format_torch(list(labelsfile)[0])
      if temp is None:
          return None
      batchidx, cls, bbox = temp
      labels_dict = {
        'batch_idx': batchidx,  # Assuming first column indicates batch_idx
        'cls': cls,  # Assuming second column is class
        'bboxes': bbox  # Assuming last columns are bbox coordinates
      }

      img_batch.requires_grad = True

      pred = trainer.model(img_batch)
      loss, _ = loss_fn(pred, labels_dict)

      trainer.model.zero_grad()
      loss.backward()

      gradient = img_batch.grad.data
      # print("GRADIENT: ", gradient)
      return gradient

# FGSM
def fgsm(image, epsilon, gradient):
    sign_data_grad = gradient.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
  
def FGSM_Attack(image_dir = '/content/custom/images', label_dir = '/content/custom/labels', epsilon = 0.1):
  gradient = ComputeGradient(image_dir, label_dir)
  if gradient is None:
    return None
  jpg_dirs = set()

  # Traverse the folder
  for root, dirs, files in os.walk(image_dir):
      for file in files:
        if file.endswith('.jpg'):
          full_path = os.path.join(root, file)
          jpg_dirs.add(full_path)

  image = load_image(list(jpg_dirs)[0])
  perturbed_image = fgsm(image, epsilon, gradient)
  return image, perturbed_image