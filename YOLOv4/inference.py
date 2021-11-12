import torch

from YOLOv4.anchor import get_anchor


def meshgrid(size, B, device):
    x = torch.arange(start=0, end=size[1], dtype=torch.float32, device=device)
    y = torch.arange(start=0, end=size[0], dtype=torch.float32, device=device)
    x, y = torch.meshgrid([x, y], indexing="ij")
    xy_grid = torch.stack(tensors=(y, x), dim=-1)
    xy_grid = torch.unsqueeze(xy_grid, dim=2)
    xy_grid = torch.unsqueeze(xy_grid, dim=0)
    xy_grid = xy_grid.repeat(B, 1, 1, 3, 1)
    return xy_grid


def encode_outputs(cfg, feature, feature_index):
    """
    编码YOLOv4的输出，生成用于loss计算的特征值
    :param feature: Tensor, shape: (batch_size, H, W, 3, C+5)
    :param feature_index: Int
    :return: Tensor, shape: (batch_size, feature_map_size, feature_map_size, 3, num_classes + 5)
    """
    anchors = get_anchor(cfg)

    shape = feature.size()
    dx_dy, dw_dh, conf, prob = torch.split(feature, [2, 2, 1, cfg["Model"]["num_classes"]], -1)

    xy_grid = meshgrid(size=shape[1:3], B=shape[0], device=cfg["device"])

    pred_xy = (torch.sigmoid(dx_dy) + xy_grid) / shape[1]
    pred_wh = torch.exp(dw_dh) * anchors[feature_index]
    pred_xywh = torch.cat(tensors=(pred_xy, pred_wh), dim=-1)
    pred_conf = torch.sigmoid(conf)
    pred_prob = torch.sigmoid(prob)
    pred_bbox = torch.cat(tensors=(pred_xywh, pred_conf, pred_prob), dim=-1)

    return pred_bbox


class Decode:
