import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

from core.SSD.anchor import DefaultBoxes
from core.SSD.loss import decode
from utils.tools import reverse_letter_box


class Decode:
    def __init__(self, cfg, original_image_size, input_image_size):
        """

        Args:
            cfg: 配置文件
            original_image_size: [h, w], 原始图片大小
            input_image_size: int, 输入SSD的图片的固定大小
        """
        self.device = cfg["device"]
        self.priors = torch.from_numpy(DefaultBoxes(cfg).__call__()).to(self.device)
        self.top_k = cfg["Decode"]["max_num_output_boxes"]
        self.num_classes = cfg["Model"]["num_classes"] + 1
        self.variance = cfg["Loss"]["variance"]
        self.conf_thresh = cfg["Decode"]["confidence_threshold"]
        self.nms_thresh = cfg["Decode"]["nms_threshold"]

        self.original_image_size = original_image_size
        self.input_image_size = input_image_size

    def __call__(self, outputs):
        loc_data, conf_data = outputs
        # loc_data: (batch_size, num_priors, 4)
        # conf_data: (batch_size, num_priors, self.num_classes)
        conf_data = F.softmax(conf_data, dim=-1)
        batch_size = loc_data.size()[0]
        num_priors = self.priors.size()[0]

        boxes_list = []
        scores_list = []
        labels_list = []

        for i in range(batch_size):
            # 对每一张图片处理
            # decoded_boxes shape: [num_priors, 4(xmin, ymin, xmax, ymax)]
            decoded_boxes = decode(loc_data[i], self.priors, self.variance)
            # 将box坐标变换到原始图片上
            decoded_boxes = reverse_letter_box(h=self.original_image_size[0], w=self.original_image_size[1],
                                               input_size=self.input_image_size, boxes=decoded_boxes)
            scores = conf_data[i]
            decoded_boxes = decoded_boxes.repeat(1, self.num_classes).reshape(scores.size()[0], -1, 4)

            # 为每一个类别创建标签信息(0~20)
            labels = torch.arange(self.num_classes, dtype=torch.int32, device=self.device)
            labels = labels.unsqueeze(0).expand_as(scores)  # shape: (num_priors, num_classes)

            # 移除背景类别的信息
            boxes = decoded_boxes[:, 1:, :]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # 对于每一个box，它都有可能属于这(num_classes-1)个类别之一
            boxes_all = boxes.reshape(-1, 4)
            scores_all = scores.reshape(-1)
            labels_all = labels.reshape(-1)

            # 移除低概率目标
            inds = torch.nonzero(scores_all > self.conf_thresh).squeeze(1)
            boxes_all, scores_all, labels_all = boxes_all[inds, :], scores_all[inds], labels_all[inds]

            boxes_all, scores_all, labels_all = boxes_all.to(torch.float32), scores_all.to(torch.float32), labels_all.to(torch.int32)

            # nms
            keep = batched_nms(boxes_all, scores_all, labels_all, iou_threshold=self.nms_thresh)
            keep = keep[:self.top_k]
            boxes_out = boxes_all[keep, :]
            scores_out = scores_all[keep]
            labels_out = labels_all[keep]
            boxes_list.append(boxes_out)
            scores_list.append(scores_out)
            labels_list.append(labels_out)
        boxes = torch.cat(tensors=boxes_list, dim=0)
        scores = torch.cat(tensors=scores_list, dim=0)
        labels = torch.cat(tensors=labels_list, dim=0) - 1
        return boxes, scores, labels

