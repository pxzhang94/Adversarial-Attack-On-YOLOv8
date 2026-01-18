import sys
sys.path.append("../")
from pathlib import Path
import time

from tutorial.run_classification import run as diagnose_classification_run
from tutorial.compute_metrics import run as compute_classification_run
from tutorial.show_diff_image_classification import run as show_classification_run
from tutorial.summary_classification import run as summary_classification_run

def run(mode, model_file, model_desc_file_no, input_folder, input_size, normalize=None, mean=None, std=None):
    diagnose_classification_run(mode=mode, model_file=model_file, input_folder=input_folder, normalize=normalize, model_desc_file_no=model_desc_file_no, input_size=input_size, mean=mean, std=std)
    compute_classification_run(input_folder=input_folder)
    show_classification_run(input_folder=input_folder)
    summary_classification_run(mode=mode, model_file=model_file, input_folder=input_folder)
    
if __name__ == "__main__":
    run(mode='upload', model_file='./vgg_cifar10_best.pt', input_folder="./demo_classification", normalize='CIFAR10', model_desc_file_no=1768547474, input_size=(32,32))    


    