import torch
import torch.nn.functional as F

from core.SSD.anchor import DefaultBoxes
from core.SSD.loss import decode
from utils.nms import diou_nms
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

        detections = torch.zeros(batch_size, self.num_classes, self.top_k, 5, dtype=torch.float32, device=self.device)
        conf_preds = torch.permute(conf_data, dims=(0, 2, 1))  # shape: (batch_size, self.num_classes, num_priors)

        # 解码
        for i in range(batch_size):
            # decoded_boxes shape: [num_priors, 4(xmin, ymin, xmax, ymax)]
            decoded_boxes = decode(loc_data[i], self.priors, self.variance)
            # 对于每个类别，应用NMS
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                # 选取满足置信度要求的boxes
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # 坐标变换到原始图片上
                boxes = reverse_letter_box(h=self.original_image_size[0], w=self.original_image_size[1],
                                           input_size=self.input_image_size, boxes=boxes)
                # 筛选出那些重叠度满足要求并且分数最大的boxes
                ids = diou_nms(boxes=boxes, scores=scores, iou_threshold=self.nms_thresh)
                count = ids.numel()
                # 保证最多不超过self.top_k个输出boxes
                n = min(count, self.top_k)
                ids = ids[:n]
                # 将检测结果concat在一起，detections最后一个维度长度是5(score, xmin, ymin, xmax, ymax)
                detections[i, cl, :n] = \
                    torch.cat((scores[ids[:n]].unsqueeze(1),
                               boxes[ids[:n]]), dim=-1)
        flt = detections.contiguous().view(batch_size, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)  # idx shape: (batch_size, self.num_classes * self.top_k)
        # rank表示这个位置的box按照score从大到小排序之后应该排在哪个位置
        _, rank = idx.sort(1)  # rank shape: (batch_size, self.num_classes * self.top_k)
        # 只保留前self.top_k个检测框
        flt[(rank >= self.top_k).unsqueeze(-1).expand_as(flt)] = 0

        pred_bboxes = torch.zeros(batch_size * self.num_classes * self.top_k, 4, dtype=torch.float32,
                                  device=self.device)
        pred_scores = torch.zeros(batch_size * self.num_classes * self.top_k, dtype=torch.float32, device=self.device)
        pred_clses = torch.zeros(batch_size * self.num_classes * self.top_k, dtype=torch.float32, device=self.device)

        pred_num = 0
        for b in range(batch_size):
            for i in range(detections.size()[1]):
                j = 0
                while detections[b, i, j, 0] >= self.conf_thresh:
                    pred_scores[pred_num] = detections[b, i, j, 0]
                    pred_bboxes[pred_num] = detections[b, i, j, 1:]
                    pred_clses[pred_num] = i - 1
                    pred_num += 1
                    j += 1

        pred_bboxes = pred_bboxes[:pred_num]
        pred_scores = pred_scores[:pred_num]
        pred_clses = pred_clses[:pred_num].to(torch.int32)
        return pred_bboxes, pred_scores, pred_clses
