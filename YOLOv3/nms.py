import torch
from torchvision.ops import nms


def gather_op(tensor, indice, device):
    """

    :param tensor: shape: (M, N)
    :param indice: shape: (K,)
    :return: Tensor, shape: (K, N)
    """
    assert tensor.dim() == 1 or tensor.dim() == 2
    if tensor.dim() == 2:
        M, N = tensor.size()
    if tensor.dim() == 1:
        M = tensor.size()[0]
        N = 1
    K = indice.size()[0]
    container = torch.zeros(K, N, dtype=torch.float32, device=device)
    for k in range(K):
        container[k] = tensor[indice[k]]
    return container


def apply_nms(cfg, boxes, scores, device):
    conf_threshold = cfg["Nms"]["conf_threshold"]
    num_classes = cfg["Model"]["num_classes"]
    iou_threshold = cfg["Nms"]["iou_threshold"]

    mask = scores >= conf_threshold

    box_list = list()
    score_list = list()
    class_list = list()

    for i in range(num_classes):
        box_of_class = boxes[mask[:, i]]
        score_of_class = scores[mask[:, i], i]
        indices = nms(boxes=box_of_class, scores=score_of_class, iou_threshold=iou_threshold)
        selected_boxes = gather_op(box_of_class, indices, device)
        selected_scores = gather_op(score_of_class, indices, device)
        select_classes = torch.ones(*selected_scores.size(), dtype=torch.int32, device=device) * i

        box_list.append(selected_boxes)
        score_list.append(selected_scores)
        class_list.append(select_classes)

    boxes = torch.cat(box_list, dim=0)
    scores = torch.cat(score_list, dim=0)
    classes = torch.cat(class_list, dim=0)

    classes = torch.squeeze(classes, dim=1)

    return boxes, scores, classes
