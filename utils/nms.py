import torch

from utils.iou import box_diou


def diou_nms(boxes, scores, iou_threshold):
    """

    :param boxes: (Tensor[N, 4]) – boxes to perform NMS on. They are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
    :param scores: (Tensor[N]) – scores for each one of the boxes
    :param iou_threshold: (float) – discards all overlapping boxes with DIoU > iou_threshold
    :return: int64 tensor with the indices of the elements that have been kept by DIoU-NMS, sorted in decreasing order of scores
    """
    order = torch.argsort(scores, dim=-1, descending=True)
    keep = list()
    while order.numel() > 0:
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break
        value = box_diou(boxes1=boxes[index], boxes2=boxes[1:])
        mask_index = torch.nonzero((value <= iou_threshold).to(dtype=torch.float32)).reshape(-1)
        order = order[mask_index + 1]
    return torch.LongTensor(keep)