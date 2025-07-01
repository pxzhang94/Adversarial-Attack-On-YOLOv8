import ultralytics
import torch
import cv2
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics import YOLO

from roboflow import Roboflow
rf = Roboflow(api_key="SyZIRlT5ZRewuvk55F78")
project = rf.workspace("vietnam-traffic-sign-detection").project("vietnam-traffic-sign-detection-2i2j8")
version = project.version(6)
dataset = version.download("yolov8")

# # Training
# args = dict(model="yolov8m.pt", data="../datasets/data.yaml", epochs=100, batch=33)
# trainer = DetectionTrainer(overrides=args)
# trainer.train()

# Testing
model = YOLO("/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/tutorial/runs/detect/train/weights/best.pt")
results = model.val(data="/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/tutorial/Vietnam-Traffic-Sign-Detection-6/data.yaml", split="test")
print(results)