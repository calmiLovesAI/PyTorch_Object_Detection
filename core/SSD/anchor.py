import itertools
from math import sqrt

import numpy as np
import torch


class DefaultBoxes:
    def __init__(self, cfg):
        self.device = cfg["device"]
        self.image_size = cfg["Train"]["input_size"]

        self.output_feature_sizes = cfg["Model"]["feature_size"]
        self.default_boxes_sizes = DefaultBoxes._get_default_boxes_sizes()
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.num_priors = len(self.aspect_ratios)
        self.steps = cfg["Model"]["downsampling_ratio"]

        self.scale_xy = 0.1
        self.scale_wh = 0.2

    @staticmethod
    def _get_default_boxes_sizes(dataset="coco"):
        if dataset == "voc":
            return [[30, 60], [60, 111], [111, 162], [162, 213], [213, 264], [264, 315]]
        elif dataset == "coco":
            return [[21, 45], [45, 99], [99, 153], [153, 207], [207, 261], [261, 315]]
        else:
            raise NotImplementedError

    def __call__(self, xyxy=True):
        f_k = self.image_size / np.array(self.steps)
        boxes = []
        for i, s in enumerate(self.output_feature_sizes):
            sk1 = self.default_boxes_sizes[i][0] / self.image_size
            sk2 = self.default_boxes_sizes[i][1] / self.image_size
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for ar in self.aspect_ratios[i]:
                w, h = sk1 * sqrt(ar), sk1 / sqrt(ar)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            for w, h in all_sizes:
                for m, n in itertools.product(range(s), repeat=2):
                    cx, cy = (n + 0.5) / f_k[i], (m + 0.5) / f_k[i]
                    boxes.append((cx, cy, w, h))
        anchors = torch.tensor(boxes, dtype=torch.float32)
        anchors = torch.clamp(anchors, min=0, max=1)
        anchors_ltrb = anchors.clone()  # (xmin, ymin, xmax, ymax)格式
        anchors_ltrb[:, 0] = anchors[:, 0] - 0.5 * anchors[:, 2]
        anchors_ltrb[:, 1] = anchors[:, 1] - 0.5 * anchors[:, 3]
        anchors_ltrb[:, 2] = anchors[:, 0] + 0.5 * anchors[:, 2]
        anchors_ltrb[:, 3] = anchors[:, 1] + 0.5 * anchors[:, 3]

        if xyxy:
            return anchors_ltrb
        else:
            return anchors
