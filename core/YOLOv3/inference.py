import numpy as np
import torch
import cv2

from core.YOLOv3.anchor import get_anchor
from core.YOLOv3.nms import apply_nms
from draw import Draw
from utils.tools import letter_box, reverse_letter_box
from torchvision.transforms.functional import to_tensor


def predict_bounding_bbox(cfg, feature_map, anchors, device, is_training=False):
    num_classes = cfg["Model"]["num_classes"]
    N, C, H, W = feature_map.size()
    feature_map = torch.permute(feature_map, dims=(0, 2, 3, 1))
    anchors = torch.reshape(anchors, shape=(1, 1, 1, -1, 2))
    grid_y = torch.reshape(torch.arange(0, H, dtype=torch.float32, device=device), (-1, 1, 1, 1))
    grid_y = torch.tile(grid_y, dims=(1, W, 1, 1))
    grid_x = torch.reshape(torch.arange(0, W, dtype=torch.float32, device=device), (1, -1, 1, 1))
    grid_x = torch.tile(grid_x, dims=(H, 1, 1, 1))
    grid = torch.cat((grid_x, grid_y), dim=-1)
    feature_map = torch.reshape(feature_map, shape=(-1, H, W, 3, num_classes + 5))
    box_xy = (torch.sigmoid(feature_map[..., 0:2]) + grid) / H
    box_wh = torch.exp(feature_map[..., 2:4]) * anchors
    confidence = torch.sigmoid(feature_map[..., 4:5])
    class_prob = torch.sigmoid(feature_map[..., 5:])
    if is_training:
        return box_xy, box_wh, grid, feature_map
    else:
        return box_xy, box_wh, confidence, class_prob


class Inference:
    def __init__(self, cfg, outputs, input_image_shape, device):
        self.cfg = cfg
        self.device = device
        self.outputs = outputs
        self.input_image_h = input_image_shape[0]
        self.input_image_w = input_image_shape[1]

    def _yolo_post_process(self, feature, scale_type):
        box_xy, box_wh, confidence, class_prob = predict_bounding_bbox(self.cfg, feature,
                                                                       get_anchor(self.cfg, scale_type, self.device),
                                                                       self.device, is_training=False)
        boxes = reverse_letter_box(self.input_image_h, self.input_image_w, self.cfg["Train"]["input_size"],
                                   torch.cat((box_xy, box_wh), dim=-1))
        boxes = torch.reshape(boxes, shape=(-1, 4))
        boxes_scores = confidence * class_prob
        boxes_scores = torch.reshape(boxes_scores, shape=(-1, self.cfg["Model"]["num_classes"]))
        return boxes, boxes_scores

    def get_results(self):
        boxes_list = list()
        boxes_scores_list = list()
        for i in range(3):
            boxes, boxes_scores = self._yolo_post_process(feature=self.outputs[i],
                                                          scale_type=i)
            boxes_list.append(boxes)
            boxes_scores_list.append(boxes_scores)
        boxes = torch.cat(boxes_list, dim=0)
        scores = torch.cat(boxes_scores_list, dim=0)
        return apply_nms(self.cfg, boxes, scores, self.device)

