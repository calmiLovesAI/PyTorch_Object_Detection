import cv2
import torch

def letter_box(image, size):
    h, w, _ = image.shape
    H, W = size
    scale = min(H / h, W / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
    top = (H - new_h) // 2
    bottom = H - new_h - top
    left = (W - new_w) // 2
    right = W - new_w - left
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return new_image, scale, [top, bottom, left, right]


def iou_2(anchors, boxes):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors)
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    anchor_max = anchors / 2
    anchor_min = - anchor_max
    box_max = boxes / 2
    box_min = - box_max
    intersect_min = torch.maximum(anchor_min, box_min)
    intersect_max = torch.minimum(anchor_max, box_max)
    intersect_wh = intersect_max - intersect_min
    intersect_wh = torch.clamp(intersect_wh, min=0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_area = boxes[..., 0] * boxes[..., 1]
    union_area = anchor_area + box_area - intersect_area
    iou = intersect_area / union_area  # shape : [N, 9]
    return iou
