import sys
sys.path.append("../")
from pathlib import Path
import time
import json
import numpy as np

from utils.util import *

noises = ['gaussian_noise', 'salt_pepper_noise', 'poisson_noise']
blurs = ['gaussian_blur', 'defocus_blur', 'motion_blur']
digital_changes = ['brightness', 'contrast', 'saturation']
adversarial_attacks = ['togf', 'togv', 'togm']
attacks = {'digital_changes': digital_changes,
           'blurs': blurs,
           'noises': noises,
           'adversarial_attacks': adversarial_attacks}

prefixes_descend = ("togf", "togv", "togm", 'gaussian_blur', 'defocus_blur', 'motion_blur', 'gaussian_noise', 'salt_pepper_noise', 'poisson_noise')
prefixes_center = ("brightness", "contrast", "saturation")
weight_descend = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
weight_center = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1])

def run(mode, model_file, input_folder):
    summary = {}
    model_name = str(Path(model_file).stem)
    summary['model_name'] = model_name
    
    report_time = time.strftime("%H:%M, %d %b %Y", time.localtime())
    summary['report_time'] = report_time
    
    if 'api' == mode:
        model_version == None
    elif 'upload' == mode:
        model_version = sha256_file(model_file)
    summary['model_version'] = model_version
    
    dataset_name = str(Path(input_folder).stem)
    summary['dataset_name'] = dataset_name
    
    dataset_version = sha256_folder(input_folder)
    summary['dataset_version'] = dataset_version
    
    with open(Path(input_folder) / 'results' / 'diagnose_results.json', 'r', encoding='utf-8') as f:
        diagnose_results = json.load(f)
    image_total = diagnose_results['original']['image_total']
    summary['image_total'] = image_total
    
    weight_no = 0
    condition_values = []
    robustness_values = []
    for condition, result in diagnose_results.items():
        if 'original' in condition:
            continue
        
        robustness = result['robustness']
        
        condition_values.append(float(condition.split("_")[-1]))
        robustness_values.append(robustness)
        weight_no += 1
        
        if weight_no == 10:
            condition_values = np.array(condition_values)
            sorted_indices = np.argsort(condition_values)
            condition_values = condition_values[sorted_indices]
            robustness_values = np.array(robustness_values)[sorted_indices]
            
            if any(condition.startswith(p) for p in prefixes_center):
                robustness_attack = np.sum(robustness_values * weight_center) / np.sum(weight_center)
            elif any(condition.startswith(p) for p in prefixes_descend):
                robustness_attack = np.sum(robustness_values * weight_descend) / np.sum(weight_descend)
            attack_name = "_".join(condition.split("_")[:-1])
            summary['robustness_' + attack_name] = round(robustness_attack, 4)
            weight_no = 0
            condition_values = []
            robustness_values = []
    
    robustness_total = 0
    for attack_group, attack_list in attacks.items():
        robustness_group = 0
        grounp_num = 0
        for attack in attack_list:
            if ('robustness_' + attack) in summary:
                robustness_group += summary['robustness_' + attack]
                grounp_num += 1
        robustness_group = robustness_group / max(1, grounp_num)
        summary['robustness_' + attack_group] = round(robustness_group, 4)
        robustness_total += robustness_group
    robustness_total = robustness_total / max(1, len(attacks))
    summary['robustness_total'] = round(robustness_total, 4)
    
    summary['summary'] = 'None'
    
    with open(str(Path(input_folder) / 'results' / 'summary.json'), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run(mode='upload', model_file='/root/autodl-tmp/project/Adversarial-Attack-On-YOLOv8/model/yolov8m.pt', input_folder="./demo_images")
    
    
        
    
            
        
    
    
    