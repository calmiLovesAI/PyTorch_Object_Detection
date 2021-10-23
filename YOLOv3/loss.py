import torch
import torch.nn.functional as F

from utils import iou_2, Iou4
from YOLOv3.inference import predict_bounding_bbox


def make_label(cfg, true_boxes):
    """

    :param cfg:
    :param true_boxes: Tensor, shape: (batch_size, N, 5)
    :return:
    """
    anchors = cfg["Train"]["anchor"]
    anchors = torch.tensor(anchors, dtype=torch.float32)
    anchors = torch.reshape(anchors, shape=(1, -1, 2))  # shape: (1, 9, 2)
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
        best_anchor_ind = torch.argmax(iou_value, dim=-1)  # shape (N,)
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
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32, device=device)
        self.anchors = torch.reshape(self.anchors, shape=(-1, 2))
        self.scale_tensor = torch.tensor(cfg["Model"]["output_features"], dtype=torch.float32)
        self.grid_shape = torch.cat((self.scale_tensor, self.scale_tensor), dim=-1)
        self.ignore_threshold = cfg["Loss"]["ignore_threshold"]

    def _get_scale_size(self, i):
        ori_size = torch.tensor([self.cfg["Train"]["input_size"], self.cfg["Train"]["input_size"]], dtype=torch.float32, device=self.device)
        anchor_size = self.anchors[i * 3:(i + 1) * 3, :]
        return torch.div(ori_size, anchor_size)

    def __call__(self, pred, target):
        total_loss = 0
        loc_loss_sum = 0
        conf_loss_sum = 0
        prob_loss_sum = 0
        B = pred[0].size()[0]

        for i in range(3):
            true_object_mask = target[i][..., 4:5]
            true_object_mask_bool = true_object_mask.bool()
            true_class_probs = target[i][..., 5:]

            pred_xy, pred_wh, grid, pred_features = predict_bounding_bbox(cfg=self.cfg,
                                                                          feature_map=pred[i],
                                                                          anchors=self.anchors,
                                                                          idx=i,
                                                                          device=self.device,
                                                                          is_training=True)
            pred_box = torch.cat((pred_xy, pred_wh), dim=-1)
            true_xy_offset = target[i][..., 0:2] * self.grid_shape[i] - grid
            true_wh_offset = torch.log(target[i][..., 2:4] * self._get_scale_size(i) + 1e-10)
            true_wh_offset = torch.where(true_object_mask_bool, true_wh_offset,
                                         torch.zeros_like(true_wh_offset, dtype=torch.float32, device=self.device))

            box_loss_scale = 2 - target[i][..., 2:3] * target[i][..., 3:4]
            ignore_mask_list = list()
            for j in range(B):
                true_box = torch.masked_select(target[i][j, ..., 0:4], true_object_mask_bool[j, ..., 0:1])
                true_box = torch.reshape(true_box, shape=(-1, 4))
                true_box = torch.unsqueeze(true_box, dim=0)
                iou = Iou4(box_1=torch.unsqueeze(pred_box[j], dim=-2), box_2=true_box).calculate_iou()
                if iou.size()[-1] != 0:
                    best_iou, _ = torch.max(iou, dim=-1)
                    ignore_mask_list.append((best_iou < self.ignore_threshold).float())
                else:
                    ignore_mask_list.append(torch.zeros(*iou.size()[:-1], dtype=torch.float32, device=self.device))

            ignore_mask = torch.stack(ignore_mask_list, dim=0)
            ignore_mask = torch.unsqueeze(ignore_mask, dim=-1)

            xy_loss = true_object_mask * box_loss_scale * F.binary_cross_entropy_with_logits(
                input=pred_features[..., 0:2],
                target=true_xy_offset,
                reduction='none')
            wh_loss = 0.5 * true_object_mask * box_loss_scale * torch.square(true_wh_offset - pred_features[..., 2:4])
            conf_loss = true_object_mask * F.binary_cross_entropy_with_logits(input=pred_features[..., 4:5],
                                                                              target=true_object_mask,
                                                                              reduction="none") + (
                                    1 - true_object_mask) * F.binary_cross_entropy_with_logits(
                input=pred_features[..., 4:5],
                target=true_object_mask,
                reduction="none") * ignore_mask
            class_loss = true_object_mask * F.binary_cross_entropy_with_logits(input=pred_features[..., 5:],
                                                                               target=true_class_probs,
                                                                               reduction="none")

            average_xy_loss = torch.sum(xy_loss) / B
            average_wh_loss = torch.sum(wh_loss) / B
            average_conf_loss = torch.sum(conf_loss) / B
            average_class_loss = torch.sum(class_loss) / B
            loc_loss_sum += average_xy_loss + average_wh_loss
            conf_loss_sum += average_conf_loss
            prob_loss_sum += average_class_loss
            total_loss += (average_xy_loss + average_wh_loss + average_conf_loss + average_class_loss)

        return total_loss, loc_loss_sum, conf_loss_sum, prob_loss_sum