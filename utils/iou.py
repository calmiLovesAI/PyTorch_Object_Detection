import torch

from torchvision.ops import box_iou


def box_iou_xywh(boxes1, boxes2):
    """
    计算box_1与box_2之间的iou
    :param box_1: Tensor, shape: (N, 4(center_x, center_y, w, h))
    :param box_2: Tensor, shape: (M, 4(center_x, center_y, w, h))
    :return: Tensor, shape: (N, M)
    """
    return box_iou(
        boxes1=torch.cat(tensors=(boxes1[:, 0:2] - 0.5 * boxes1[:, 2:4], boxes1[:, 0:2] + 0.5 * boxes1[:, 2:4]),
                         dim=-1),
        boxes2=torch.cat(tensors=(boxes2[:, 0:2] - 0.5 * boxes2[:, 2:4], boxes2[:, 0:2] + 0.5 * boxes2[:, 2:4]),
                         dim=-1))


def box_ciou(boxes1, boxes2):
    """
    计算boxes1与boxes2之间的ciou值
    :param boxes1: Tensor, shape: (..., 4 (xmin, ymin, xmax, ymax))
    :param boxes2: Tensor, shape: (..., 4 (xmin, ymin, xmax, ymax))
    :return:
    """
    box_1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    box_2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    intersect_min = torch.maximum(boxes1[..., 0:2], boxes2[..., 0:2])
    intersect_max = torch.minimum(boxes1[..., 2:4], boxes2[..., 2:4])
    intersect_wh = intersect_max - intersect_min
    intersect_wh = torch.clamp(intersect_wh, min=0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    union_area = box_1_area + box_2_area - intersect_area
    iou = intersect_area / union_area



def box_ciou_xywh(boxes1, boxes2):
    """
    计算boxes1与boxes2之间的ciou值
    :param boxes1: Tensor, shape: (..., 4 (cx, cy, w, h))
    :param boxes2: Tensor, shape: (..., 4 (cx, cy, w, h))
    :return:
    """
    return box_ciou(
        boxes1=torch.cat(tensors=(boxes1[:, 0:2] - 0.5 * boxes1[:, 2:4], boxes1[:, 0:2] + 0.5 * boxes1[:, 2:4]),
                         dim=-1),
        boxes2=torch.cat(tensors=(boxes2[:, 0:2] - 0.5 * boxes2[:, 2:4], boxes2[:, 0:2] + 0.5 * boxes2[:, 2:4]),
                         dim=-1))
