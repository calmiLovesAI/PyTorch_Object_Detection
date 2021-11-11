import torch

from YOLOv4.anchor import get_anchor
from YOLOv4.inference import encode_outputs
from utils.iou import box_iou_xywh


def make_label(cfg, target):
    input_size = cfg["Train"]["input_size"]
    num_classes = cfg["Model"]["num_classes"]
    device = cfg["device"]

    output_feature_sizes = [input_size // i for i in cfg["Model"]["yolo_strides"]]
    strides = torch.tensor(cfg["Model"]["yolo_strides"], dtype=torch.float32, device=device)
    anchors = get_anchor(cfg)

    batch_size = target.size()[0]
    batch_labels = [
        torch.zeros(batch_size, fs, fs, 3, 5 + num_classes, dtype=torch.float32, device=device)
        for fs in output_feature_sizes]

    for i in range(batch_size):
        true_boxes = target[i]
        true_boxes = true_boxes[true_boxes[..., -1] != -1]
        for n in range(true_boxes.size()[0]):
            box_xyxy = true_boxes[i, :4]
            box_xywh = torch.cat(tensors=((box_xyxy[:2] + box_xyxy[2:]) * 0.5, box_xyxy[2:] - box_xyxy[:2]), dim=-1)
            scaled_box_xywh = torch.unsqueeze(box_xywh, dim=0) * input_size / torch.unsqueeze(strides,
                                                                                              dim=1)  # shape : (3, 4)
            box_class = true_boxes[i, 4].to(dtype=torch.int32)
            one_hot = torch.zeros(num_classes, dtype=torch.float32, device=device)
            one_hot[box_class] = 1.0

            iou_list = []
            for j in range(3):
                anchors_xywh = torch.zeros(3, 4, dtype=torch.float32, device=device)
                anchors_xywh[:, 2:4] = anchors[j] / output_feature_sizes[j]

                zero_xy_scaled_box = torch.zeros(3, 4, dtype=torch.float32, device=device)
                zero_xy_scaled_box[:, 2:4] = scaled_box_xywh[:, 2:4]

                iou = box_iou_xywh(boxes1=zero_xy_scaled_box[j], boxes2=anchors_xywh)
                iou_list.append(iou)

            iou_tensor = torch.stack(tensors=iou_list, dim=0)  # shape: (3, 3)
            best_anchor_ind = torch.argmax(iou_tensor.reshape(-1), dim=-1)

            level_idx = best_anchor_ind // 3
            anchor_idx = best_anchor_ind % 3

            x_ind, y_ind = torch.floor(scaled_box_xywh[level_idx, 0:2]).to(torch.int32)
            batch_labels[level_idx][i, y_ind, x_ind, anchor_idx, :] = 0
            batch_labels[level_idx][i, y_ind, x_ind, anchor_idx, 0:4] = box_xywh
            batch_labels[level_idx][i, y_ind, x_ind, anchor_idx, 4:5] = 1.0
            batch_labels[level_idx][i, y_ind, x_ind, anchor_idx, 5:] = one_hot

    return batch_labels


class YoloLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]
        self.strides = cfg["Model"]["yolo_strides"]
        self.ignore_threshold = cfg["Loss"]["ignore_threshold"]

    def __call__(self, pred, target):
        ciou_loss = 0
        conf_loss = 0
        prob_loss = 0
        for i in range(3):
            label = target[i]
            feature = pred[i]
            N, C, H, W = feature.size()
            feature = torch.permute(feature, dims=(0, 2, 3, 1))
            feature_reshaped = torch.reshape(feature, (N, H, W, 3, -1))
            raw_xywh = feature_reshaped[..., 0:4]
            raw_conf = feature_reshaped[..., 4:5]
            raw_prob = feature_reshaped[..., 5:]
            label_xywh = label[..., 0:4]
            label_conf = label[..., 4:5]
            label_prob = label[..., 5:]

            pred_bbox = encode_outputs(self.cfg, feature_reshaped, i)
            pred_xywh = pred_bbox[..., 0:4]

            bbox_loss_scale = 2.0 - label_xywh[..., 2:3] * label_xywh[..., 3:4]
            ciou = torch.unsqueeze(CIoU(box_1=pred_xywh, box_2=label_xywh).calculate_ciou(), dim=-1)
            ciou_loss = label_conf * bbox_loss_scale * (1 - ciou)
