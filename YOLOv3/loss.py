import torch

from utils import iou_2


def make_label(cfg, true_boxes):
    """

    :param cfg:
    :param true_boxes: Tensor, shape: (batch_size, N, 5)
    :return:
    """
    anchors = cfg["Train"]["anchor"]
    anchors = torch.tensor(anchors, dtype=torch.float32)
    anchors = torch.reshape(anchors, shape=(1, -1, 2))   # shape: (1, 9, 2)
    anchor_index = cfg["Train"]["anchor_index"]
    features_size = cfg["Model"]["output_features"]
    num_classes = cfg["Model"]["num_classes"]
    batch_size = true_boxes.size()[0]

    center_xy = torch.div(true_boxes[..., 0:2] + true_boxes[..., 2:4], 2)  # shape : [B, N, 2]
    box_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]  # shape : [B, N, 2]
    true_labels = [torch.zeros(batch_size, features_size[i], features_size[i], 3, num_classes + 5) for i in range(3)]
    valid_mask = box_wh[..., 0] > 0

    for b in range(batch_size):
        wh = box_wh[b, valid_mask[b]]
        if wh.size()[0] == 0:
            continue
        wh = torch.unsqueeze(wh, dim=1)  # shape: (N, 1, 2)
        iou_value = iou_2(anchors, wh)
        best_anchor_ind = torch.argmax(iou_value, dim=-1) # shape (N,)
        for i, n in enumerate(best_anchor_ind):
            for s in range(3):
                if n in anchor_index[s]:
                    x = torch.floor(true_boxes[b, i, 0] * features_size[s]).int()
                    y = torch.floor(true_boxes[b, i, 1] * features_size[s]).int()
                    anchor_id = anchor_index[s].index(n)
                    class_id = true_boxes[b, i, -1].int()
                    true_labels[s][b, y, x, anchor_id, 0:4] = true_boxes[b, i, 0:4]
                    true_labels[s][b, y, x, anchor_id, 4] = 1
                    true_labels[s][b, y, x, anchor_id, 5 + class_id] = 1

    return true_labels


