import numpy as np
import torch
import os

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

def read_yolo_labels(txt_path):
    """Return list of (cls, x, y, w, h) in normalized YOLO format."""
    if not os.path.exists(txt_path):
        return []
    boxes = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5 and len(parts) != 6:
                # 容忍可能有 score，一并处理
                # 格式: cls x y w h [score]
                pass
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
                boxes.append((cls, x, y, w, h))
            except Exception:
                continue
    return boxes

def write_yolo_labels(txt_path, boxes):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
def rotate_yolo_path(label_path, angle, img_width, img_height):
    bboxes = load_yolo_bboxes_from_txt(label_path)
    return rotate_yolo_bboxes(bboxes, angle, img_width, img_height)

def rotate_yolo_bboxes(bboxes, angle, img_width, img_height):
    """
    旋转 YOLO 格式的目标框，绕图像中心旋转。
    :param bboxes: list of [class_id, cx, cy, w, h] (cx, cy, w, h 都是归一化坐标)
    :param angle_deg: 旋转角度（逆时针，单位：度）
    :param img_width: 图像宽度（像素）
    :param img_height: 图像高度（像素）
    :return: 旋转后的 bboxes（保持 YOLO 格式）
    """

    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    cx_img, cy_img = img_width / 2, img_height / 2

    rotated_bboxes = []

    for bbox in bboxes:
        class_id, cx, cy, w, h = bbox
        # 恢复到像素空间
        bw, bh = w * img_width, h * img_height
        bx, by = cx * img_width, cy * img_height

        # 四个角点
        corners = np.array([
            [bx - bw/2, by - bh/2],
            [bx + bw/2, by - bh/2],
            [bx + bw/2, by + bh/2],
            [bx - bw/2, by + bh/2]
        ])

        # 绕图像中心旋转
        rotated_corners = []
        for x, y in corners:
            dx, dy = x - cx_img, y - cy_img
            new_x = cos_a * dx - sin_a * dy + cx_img
            new_y = sin_a * dx + cos_a * dy + cy_img
            rotated_corners.append([new_x, new_y])
        rotated_corners = np.array(rotated_corners)

        # 找旋转后的最小包围框
        x_min = np.clip(np.min(rotated_corners[:, 0]), 0, img_width)
        x_max = np.clip(np.max(rotated_corners[:, 0]), 0, img_width)
        y_min = np.clip(np.min(rotated_corners[:, 1]), 0, img_height)
        y_max = np.clip(np.max(rotated_corners[:, 1]), 0, img_height)

        # 新的中心与宽高（像素）
        new_bw = x_max - x_min
        new_bh = y_max - y_min
        new_bx = (x_min + x_max) / 2
        new_by = (y_min + y_max) / 2

        # 如果bbox太小或裁剪掉了，跳过
        if new_bw <= 1 or new_bh <= 1:
            continue

        # 归一化后添加
        rotated_bboxes.append([
            class_id,
            round(new_bx / img_width, 6),
            round(new_by / img_height, 6),
            round(new_bw / img_width, 6),
            round(new_bh / img_height, 6)
        ])

    return rotated_bboxes

def yolo_to_xyxy(box, W, H):
    """
    YOLO 归一化 -> 绝对像素 (x1,y1,x2,y2)，并限制在图像边界内。
    返回: (cls, x1, y1, x2, y2)
    """
    cls, x, y, w, h = box
    cx, cy = x * W, y * H
    bw, bh = w * W, h * H
    x1, y1 = int(round(cx - bw / 2)), int(round(cy - bh / 2))
    x2, y2 = int(round(cx + bw / 2)), int(round(cy + bh / 2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    return cls, x1, y1, x2, y2