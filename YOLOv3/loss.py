import torch

from utils import iou_2
from YOLOv3.inference import predict_bounding_bbox


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

    center_xy = torch.div(true_boxes[..., 0:2] + true_boxes[..., 2:4], 2, rounding_mode="floor")  # shape : [B, N, 2]
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


class YoloLoss:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.anchors = cfg["Train"]["anchor"]
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32)
        self.anchors = torch.reshape(self.anchors, shape=(-1, 2))
        self.scale_tensor = torch.tensor(cfg["Model"]["output_features"], dtype=torch.float32)
        self.grid_shape = torch.cat((self.scale_tensor, self.scale_tensor), dim=-1)

    def __call__(self, pred, target):
        total_loss = 0
        B = pred[0].size()[0]

        for i in range(3):
            true_object_mask = target[..., 4:5]
            true_object_mask_bool = true_object_mask.bool()
            true_class_probs = target[i][..., 5:]

            pred_xy, pred_wh, grid, pred_features = predict_bounding_bbox(cfg=self.cfg,
                                                                          feature_map=pred[i],
                                                                          anchors=self.anchors,
                                                                          idx=i,
                                                                          device=self.device,
                                                                          is_training=True)
            pred_box = torch.cat((pred_xy, pred_wh), dim=-1)

