# ---------------------------------
# BBox convert & IoU / matching
# ---------------------------------
def yolo_to_xyxy(box, W, H):
    cls, x, y, w, h = box
    cx, cy = x * W, y * H
    bw, bh = w * W, h * H
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2
    return cls, max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)

def xyxy_to_yolo(box, W, H):
    cls, x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = x1 + bw / 2, y1 + bh / 2
    return cls, cx / W, cy / H, bw / W, bh / H

def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union

def match_and_score_single(gts_yolo, preds_yolo, W, H, iou_thresh = 0.5):
    """
    返回:
      image_correct: bool
      object_total: int
      object_correct: int
    匹配策略：贪心按 IoU 从大到小匹配，类别需相同，IoU>=阈值。
    """
    gts_xy = [(cls, *yolo_to_xyxy((cls, x, y, w, h), W, H)[1:]) for cls, x, y, w, h in gts_yolo]
    preds_xy = [(cls, *yolo_to_xyxy((cls, x, y, w, h), W, H)[1:]) for cls, x, y, w, h in preds_yolo]

    object_total = len(gts_xy)
    object_correct = 0

    if object_total == 0:
        # 无 GT：定义为只有在无预测时图像正确；object_total=0 不计入分母
        image_correct = (len(preds_xy) == 0)
        return image_correct, 0, 0

    used_pred = set()
    # 计算所有 (gt, pred) 的 IoU 并排序
    pairs = []
    for gi, (gcls, gx1, gy1, gx2, gy2) in enumerate(gts_xy):
        for pi, (pcls, px1, py1, px2, py2) in enumerate(preds_xy):
            iou = box_iou_xyxy((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
            pairs.append((iou, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_gt = [False] * len(gts_xy)
    for iou, gi, pi in pairs:
        if iou < iou_thresh or pi in used_pred or matched_gt[gi]:
            continue
        if gts_xy[gi][0] == preds_xy[pi][0]:  # 类别必须一致
            matched_gt[gi] = True
            used_pred.add(pi)
            object_correct += 1

    image_correct = (object_correct == object_total)
    return image_correct, object_total, object_correct