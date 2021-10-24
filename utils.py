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
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                                   value=(128, 128, 128))
    return new_image, scale, [top, bottom, left, right]


class ResizeWithPad:
    def __init__(self, cfg, h, w):
        super(ResizeWithPad, self).__init__()
        self.H = cfg["Train"]["input_size"]
        self.W = cfg["Train"]["input_size"]
        self.h = h
        self.w = w

    def get_transform_coefficient(self):
        if self.h <= self.w:
            longer_edge = "w"
            scale = self.W / self.w
            padding_length = (self.H - self.h * scale) / 2
        else:
            longer_edge = "h"
            scale = self.H / self.h
            padding_length = (self.W - self.w * scale) / 2
        return longer_edge, scale, padding_length

    def raw_to_resized(self, x_min, y_min, x_max, y_max):
        longer_edge, scale, padding_length = self.get_transform_coefficient()
        x_min = x_min * scale
        x_max = x_max * scale
        y_min = y_min * scale
        y_max = y_max * scale
        if longer_edge == "h":
            x_min += padding_length
            x_max += padding_length
        else:
            y_min += padding_length
            y_max += padding_length
        return x_min, y_min, x_max, y_max

    def resized_to_raw(self, center_x, center_y, width, height):
        longer_edge, scale, padding_length = self.get_transform_coefficient()
        center_x *= self.W
        width *= self.W
        center_y *= self.H
        height *= self.H
        if longer_edge == "h":
            center_x -= padding_length
        else:
            center_y -= padding_length
        center_x = center_x / scale
        center_y = center_y / scale
        width = width / scale
        height = height / scale
        return center_x, center_y, width, height


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


class Iou4:
    def __init__(self, box_1, box_2):
        """

        :param box_1: Tensor, shape: (..., 4(cx, cy, w, h))
        :param box_2: Tensor, shape: (..., 4(cx, cy, w, h))
        """
        self.box_1_min, self.box_1_max = Iou4._get_box_min_and_max(box_1)
        self.box_2_min, self.box_2_max = Iou4._get_box_min_and_max(box_2)
        self.box_1_area = box_1[..., 2] * box_1[..., 3]
        self.box_2_area = box_2[..., 2] * box_2[..., 3]

    @staticmethod
    def _get_box_min_and_max(box):
        box_xy = box[..., 0:2]
        box_wh = box[..., 2:4]
        box_min = box_xy - box_wh / 2
        box_max = box_xy + box_wh / 2
        return box_min, box_max

    def calculate_iou(self):
        intersect_min = torch.maximum(self.box_1_min, self.box_2_min)
        intersect_max = torch.minimum(self.box_1_max, self.box_2_max)
        intersect_wh = intersect_max - intersect_min
        intersect_wh = torch.clamp(intersect_wh, min=0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = self.box_1_area + self.box_2_area - intersect_area
        iou = intersect_area / union_area
        return iou


class MeanMetric:
    def __init__(self):
        self.accumulated = 0
        self.count = 0

    def update(self, value):
        self.accumulated += value
        self.count += 1

    def result(self):
        return self.accumulated / self.count

    def reset(self):
        self.__init__()
