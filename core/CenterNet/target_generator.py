import torch

from core.CenterNet.gaussian import gaussian_radius, draw_umich_gaussian


class TargetGenerator:
    def __init__(self, cfg, batch_labels):
        self.device = cfg["device"]
        self.input_size = cfg["Train"]["input_size"]
        self.downsampling_ratio = cfg["Model"]["downsampling_ratio"]
        self.features_shape = torch.tensor(
            data=[self.input_size // self.downsampling_ratio, self.input_size // self.downsampling_ratio],
            dtype=torch.int32, device=self.device)
        self.batch_labels = batch_labels
        self.batch_size = batch_labels.size()[0]
        self.max_num_boxes = cfg["Train"]["max_num_boxes"]
        self.num_classes = cfg["Model"]["num_classes"]

    def __call__(self, *args, **kwargs):
        gt_heatmap = torch.zeros(self.batch_size, self.features_shape[0], self.features_shape[1], self.num_classes,
                                 dtype=torch.float32, device=self.device)
        gt_reg = torch.zeros(self.batch_size, self.max_num_boxes, 2, dtype=torch.float32, device=self.device)
        gt_wh = torch.zeros(self.batch_size, self.max_num_boxes, 2, dtype=torch.float32, device=self.device)
        gt_reg_mask = torch.zeros(self.batch_size, self.max_num_boxes, dtype=torch.float32, device=self.device)
        gt_indices = torch.zeros(self.batch_size, self.max_num_boxes, dtype=torch.float32, device=self.device)
        for i, label in enumerate(self.batch_labels):
            label = label[label[:, -1] != -1]    # shape: (N, 5)
            hm, reg, wh, reg_mask, ind = self._parse_label(label)
            gt_heatmap[i, :, :, :] = hm
            gt_reg[i, :, :] = reg
            gt_wh[i, :, :] = wh
            gt_reg_mask[i, :] = reg_mask
            gt_indices[i, :] = ind
        return gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices

    def _parse_label(self, label):
        hm = torch.zeros(self.features_shape[0], self.features_shape[1], self.num_classes, dtype=torch.float32,
                         device=self.device)
        reg = torch.zeros(self.max_num_boxes, 2, dtype=torch.float32, device=self.device)
        wh = torch.zeros(self.max_num_boxes, 2, dtype=torch.float32, device=self.device)
        reg_mask = torch.zeros(self.max_num_boxes, dtype=torch.float32, device=self.device)
        ind = torch.zeros(self.max_num_boxes, dtype=torch.float32, device=self.device)
        for j, item in enumerate(label):
            item[:4] = item[:4] * self.input_size / self.downsampling_ratio
            xmin, ymin, xmax, ymax, class_id = item
            class_id = class_id.to(dtype=torch.int32)
            h, w = int(ymax - ymin), int(xmax - xmin)
            radius = gaussian_radius((h, w))
            radius = max(0, int(radius))
            ctr_x, ctr_y = (xmin + xmax) / 2, (ymin + ymax) / 2
            center_point = torch.tensor([ctr_x, ctr_y], dtype=torch.float32)
            center_point_int = center_point.to(dtype=torch.int32)
            _hm = draw_umich_gaussian(hm[:, :, class_id], center_point_int, radius)
            hm[:, :, class_id] = torch.from_numpy(_hm).to(self.device)

            reg[j] = center_point - center_point_int
            wh[j] = torch.tensor(data=[w, h], dtype=torch.float32, device=self.device)
            reg_mask[j] = 1
            ind[j] = center_point_int[1] * self.features_shape[1] + center_point_int[0]
        return hm, reg, wh, reg_mask, ind
