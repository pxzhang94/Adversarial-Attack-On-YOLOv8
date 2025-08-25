import numpy as np
import cv2
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def compute_metrics(img1, img2):
    """
    计算 L2, L∞ 和 SSIM 指标
    输入:
        img1, img2: 读入的两张图像 (numpy array, HxWxC)
    返回:
        dict: {"L2": ..., "Linf": ..., "SSIM": ...}
    """

    # 确保两张图像大小相同
    assert img1.shape == img2.shape, "两张图像尺寸不一致！"

    # 转换为浮点型
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 计算差值
    diff = img1 - img2

    # L2 范数 (均方根误差 RMSE)
    l2 = np.sqrt(np.mean(diff ** 2))

    # L∞ 范数 (最大绝对误差)
    linf = np.max(np.abs(diff))

    # SSIM (结构相似性)
    # 只支持单通道，所以转为灰度
    img1_gray = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    ssim_value, _ = ssim(img1_gray, img2_gray, full=True)

    return [l2, linf, ssim_value]


def run(input_folder):
    folder = Path(input_folder) / 'results'
    transformation_paths = sorted([p for p in folder.iterdir() if p.is_dir()])
    # result_dict = {}
    result_array = []
    for transformation_path in transformation_paths:
        image_paths = sorted([p for p in (transformation_path / 'images').iterdir() if p.suffix.lower() == ".jpg"])
        metric = np.asarray([0.0, 0.0, 0.0])
        for image_path in image_paths:
            adversarial_img = cv2.imread(str(image_path))
            
            original_path = Path(input_folder) / 'images' / image_path.name
            original_img = cv2.imread(str(original_path))
            
            adversarial_img = cv2.cvtColor(adversarial_img, cv2.COLOR_BGR2RGB)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            metric += compute_metrics(original_img, adversarial_img)
        metric = metric / len(image_paths)
        # result = {'L2': metric[0], 'Linf': metric[1], 'SSIM': metric[2]}
        # result_dict[transformation_path.stem] = result
        result_array.append([str(transformation_path).split('/')[-1], metric[0], metric[1], metric[2]])
    print(result_array)
    # with open(str(folder / 'metrics.json'), "w", encoding="utf-8") as f:
    #     json.dump(result_dict, f, ensure_ascii=False, indent=4)
    
    # 转换为 DataFrame
    df = pd.DataFrame(result_array, columns=["Condition", "L2", "Linf", "SSIM"])
    # 保存为 CSV
    df.to_csv(str(folder / 'metrics.csv'), index=False)
    
if __name__ == "__main__":
    run("./demo_images")
