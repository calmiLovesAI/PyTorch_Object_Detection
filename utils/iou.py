import math

import torch

eps = 1e-6


def box_iou(boxes1, boxes2):
    """

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
    iou = intersect_area / torch.clamp(union_area, min=eps)
    return iou


def box_iou_xywh(boxes1, boxes2):
    """
    计算boxes1与boxes2之间的iou
    :param box_1: Tensor, shape: (..., 4(center_x, center_y, w, h))
    :param box_2: Tensor, shape: (..., 4(center_x, center_y, w, h))
    :return:
    """
    return box_iou(
        boxes1=torch.cat(tensors=(boxes1[..., 0:2] - 0.5 * boxes1[..., 2:4], boxes1[..., 0:2] + 0.5 * boxes1[..., 2:4]),
                         dim=-1),
        boxes2=torch.cat(tensors=(boxes2[..., 0:2] - 0.5 * boxes2[..., 2:4], boxes2[..., 0:2] + 0.5 * boxes2[..., 2:4]),
                         dim=-1))


def box_diou(boxes1, boxes2):
    """
    计算boxes1与boxes2之间的diou
    :param boxes1: Tensor, shape: (..., 4 (xmin, ymin, xmax, ymax))
    :param boxes2: Tensor, shape: (..., 4 (xmin, ymin, xmax, ymax))
    :return:
    """
    box_1_xywh = torch.cat(tensors=((boxes1[..., 0:2] + boxes1[..., 2:4]) / 2, boxes1[..., 2:4] - boxes1[..., 0:2]),
                           dim=-1)
    box_2_xywh = torch.cat(tensors=((boxes2[..., 0:2] + boxes2[..., 2:4]) / 2, boxes2[..., 2:4] - boxes2[..., 0:2]),
                           dim=-1)

    iou = box_iou(boxes1, boxes2)

    # 闭包
    enclose_left_up = torch.minimum(boxes1[..., 0:2], boxes2[..., 0:2])
    enclose_right_down = torch.maximum(boxes1[..., 2:4], boxes2[..., 2:4])
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_wh = torch.clamp(enclose_wh, min=0)
    enclose_c_square = torch.sum(torch.pow(enclose_wh[..., 0:2], 2), dim=-1)  # 闭包区域的对角线距离的平方
    d_square = torch.sum(torch.pow(box_1_xywh[..., 0:2] - box_2_xywh[..., 0:2], 2), dim=-1)  # 中心点之间的距离的平方

    diou = iou - d_square / torch.clamp(enclose_c_square, min=eps)
    return diou


def box_ciou(boxes1, boxes2):
    """
    计算boxes1与boxes2之间的ciou，boxes1是预测值，boxes2是真实值
    :param boxes1: Tensor, shape: (..., 4 (xmin, ymin, xmax, ymax))
    :param boxes2: Tensor, shape: (..., 4 (xmin, ymin, xmax, ymax))
    :return:
    """

    box_1_xywh = torch.cat(tensors=((boxes1[..., 0:2] + boxes1[..., 2:4]) / 2, boxes1[..., 2:4] - boxes1[..., 0:2]),
                           dim=-1)
    box_2_xywh = torch.cat(tensors=((boxes2[..., 0:2] + boxes2[..., 2:4]) / 2, boxes2[..., 2:4] - boxes2[..., 0:2]),
                           dim=-1)

    iou = box_iou(boxes1, boxes2)

    # 闭包
    enclose_left_up = torch.minimum(boxes1[..., 0:2], boxes2[..., 0:2])
    enclose_right_down = torch.maximum(boxes1[..., 2:4], boxes2[..., 2:4])
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_wh = torch.clamp(enclose_wh, min=0)
    enclose_c_square = torch.sum(torch.pow(enclose_wh[..., 0:2], 2), dim=-1)  # 闭包区域的对角线距离的平方
    d_square = torch.sum(torch.pow(box_1_xywh[..., 0:2] - box_2_xywh[..., 0:2], 2), dim=-1)  # 中心点之间的距离的平方

    v = (4.0 / math.pi ** 2) * torch.pow(
        torch.atan(box_2_xywh[..., 2] / torch.clamp(box_2_xywh[..., 3], min=eps)) - torch.atan(
            box_1_xywh[..., 2] / torch.clamp(box_1_xywh[..., 3], min=eps)), 2)
    with torch.no_grad():
        alpha = v / torch.clamp(1.0 - iou + v, min=eps)

    ciou = iou - d_square / torch.clamp(enclose_c_square, min=eps) - alpha * v
    return ciou


def box_ciou_xywh(boxes1, boxes2):
    """
    计算boxes1与boxes2之间的ciou值
    :param boxes1: Tensor, shape: (..., 4 (cx, cy, w, h))
    :param boxes2: Tensor, shape: (..., 4 (cx, cy, w, h))
    :return:
    """
    return box_ciou(
        boxes1=torch.cat(tensors=(boxes1[..., 0:2] - 0.5 * boxes1[..., 2:4], boxes1[..., 0:2] + 0.5 * boxes1[..., 2:4]),
                         dim=-1),
        boxes2=torch.cat(tensors=(boxes2[..., 0:2] - 0.5 * boxes2[..., 2:4], boxes2[..., 0:2] + 0.5 * boxes2[..., 2:4]),
                         dim=-1))


def box_giou(boxes1, boxes2):
    """
    计算boxes1与boxes2之间的ciou，boxes1是预测值，boxes2是真实值
    :param boxes1: Tensor, shape: (..., 4 (xmin, ymin, xmax, ymax))
    :param boxes2: Tensor, shape: (..., 4 (xmin, ymin, xmax, ymax))
    :return:
    """
    # iou
    box_1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    box_2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    intersect_min = torch.maximum(boxes1[..., 0:2], boxes2[..., 0:2])
    intersect_max = torch.minimum(boxes1[..., 2:4], boxes2[..., 2:4])
    intersect_wh = intersect_max - intersect_min
    intersect_wh = torch.clamp(intersect_wh, min=0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    union_area = box_1_area + box_2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=eps)

    # 闭包
    enclose_left_up = torch.minimum(boxes1[..., 0:2], boxes2[..., 0:2])
    enclose_right_down = torch.maximum(boxes1[..., 2:4], boxes2[..., 2:4])
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_wh = torch.clamp(enclose_wh, min=0)
    # 闭包面积
    area_enclose = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou = iou - (area_enclose - union_area) / torch.clamp(area_enclose, min=eps)
    return torch.clamp(giou, min=-1.0, max=1.0)


def box_giou_xywh(boxes1, boxes2):
    """
    计算boxes1与boxes2之间的ciou值
    :param boxes1: Tensor, shape: (..., 4 (cx, cy, w, h))
    :param boxes2: Tensor, shape: (..., 4 (cx, cy, w, h))
    :return:
    """
    return box_giou(
        boxes1=torch.cat(tensors=(boxes1[..., 0:2] - 0.5 * boxes1[..., 2:4], boxes1[..., 0:2] + 0.5 * boxes1[..., 2:4]),
                         dim=-1),
        boxes2=torch.cat(tensors=(boxes2[..., 0:2] - 0.5 * boxes2[..., 2:4], boxes2[..., 0:2] + 0.5 * boxes2[..., 2:4]),
                         dim=-1))


# From: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)
