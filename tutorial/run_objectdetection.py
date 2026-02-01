import sys
sys.path.append("../")
from pathlib import Path
import time

from tutorial.diagnose_objectdetection import run as diagnose_objectdetection_run
from tutorial.compute_metrics import run as compute_objectdetection_run
from tutorial.show_diff_image_objectdetection import run as show_objectdetection_run
from tutorial.summary_objectdetection import run as summary_objectdetection_run

def run(mode, model_file, model_desc_file_no, input_folder, input_size, normalize=None, mean=None, std=None, device = 'cuda' if torch.cuda.is_available() else 'cpu', conf_thresh=0.3, iou_thresh=0.5):
    diagnose_objectdetection_run(mode=mode, model_file=model_file, input_folder=input_folder, normalize=normalize, model_desc_file_no=model_desc_file_no, input_size=input_size, mean=mean, std=std, device=device, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    compute_objectdetection_run(input_folder=input_folder)
    show_objectdetection_run(input_folder=input_folder)
    summary_objectdetection_run(mode=mode, model_file=model_file, input_folder=input_folder)
    
if __name__ == "__main__":
    run(mode='upload', model_file='/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/model/yolov8m.pt', input_folder="./demo_images", model_desc_file_no=1768547475, input_size=(640,640), device = 'cuda' if torch.cuda.is_available() else 'cpu', conf_thresh=0.3, iou_thresh=0.5)    


    