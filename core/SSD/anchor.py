import numpy as np
import math

from itertools import product

from core.SSD.aspect_ratio import get_aspect_ratio


class DefaultBoxes:
    def __init__(self, cfg):
        self.device = cfg["device"]
        self.image_size = cfg["Train"]["input_size"]

        self.output_feature_sizes = cfg["Model"]["feature_size"]
        self.default_boxes_sizes = cfg["Model"]["default_boxes_sizes"]
        self.aspect_ratios = get_aspect_ratio(cfg)
        self.num_priors = len(self.aspect_ratios)
        self.steps = cfg["Model"]["downsampling_ratio"]

    def __call__(self, *args, **kwargs):
        boxes = []
        for k, f in enumerate(self.output_feature_sizes):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # box中心点的坐标
                center_x = (j + 0.5) / f_k
                center_y = (i + 0.5) / f_k
                # box的高和宽
                s_min = self.default_boxes_sizes[k][0] / self.image_size
                s_max = math.sqrt(self.default_boxes_sizes[k][0] * self.default_boxes_sizes[k][1]) / self.image_size
                boxes += [center_x, center_y, s_min, s_min]
                boxes += [center_x, center_y, s_max, s_max]
                for ar in self.aspect_ratios[k]:
                    boxes += [center_x, center_y, s_min * math.sqrt(ar), s_min / math.sqrt(ar)]

        output = np.array(boxes).reshape((-1, 4))
        output = np.clip(a=output, a_min=0, a_max=1)
        return output

