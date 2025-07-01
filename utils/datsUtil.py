import numpy as np
import torch

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