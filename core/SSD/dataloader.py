import torch

from core.SSD.anchor import DefaultBoxes
from core.SSD.loss import jaccard
from dataset.public_dataloader import PublicTrainLoader


class UpdateClassIndices:
    def __call__(self, image, target):
        target[..., -1] += 1
        return image, target


class AssignGTToDefaultBoxes:
    def __init__(self, cfg):
        self.default_boxes = DefaultBoxes(cfg).__call__(xyxy=True)  # shape: (8732, 4)
        self.threshold = cfg["Loss"]["overlap_thresh"]

    def __call__(self, image, target):
        """

        Args:
            image:
            target: torch.Tensor, shape: (N, 5)

        Returns:

        """
        boxes = target[:, :-1]
        labels_in = target[:, -1].long()
        overlaps = jaccard(boxes, self.default_boxes)
        # 每个default_box对应的最大IoU值的gt_box
        best_dbox_ious, best_dbox_idx = overlaps.max(dim=0)   # [8732]
        # 每个gt_box对应的最大IoU值的default_box
        best_bbox_ious, best_bbox_idx = overlaps.max(dim=1)   # [N]

        # 将每个gt匹配到的最佳default_box设置为正样本
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)
        idx = torch.arange(0, best_bbox_idx.size(dim=0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # 将与gt的IoU大于给定阈值的default_boxes设置为正样本
        masks = best_dbox_ious > self.threshold
        labels_out = torch.zeros(self.default_boxes.size(0), dtype=torch.int64)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]

        # 将default_box匹配到正样本的位置设置成对应的gt信息
        bboxes_out = self.default_boxes.clone()
        bboxes_out[masks, :] = boxes[best_dbox_idx[masks], :]

        cx = (bboxes_out[:, 0] + bboxes_out[:, 2]) / 2
        cy = (bboxes_out[:, 1] + bboxes_out[:, 3]) / 2
        w = bboxes_out[:, 2] - bboxes_out[:, 0]
        h = bboxes_out[:, 3] - bboxes_out[:, 1]
        bboxes_out[:, 0] = cx
        bboxes_out[:, 1] = cy
        bboxes_out[:, 2] = w
        bboxes_out[:, 3] = h

        target_out = torch.cat(tensors=(bboxes_out, labels_out.unsqueeze(-1)), dim=-1)

        return image, target_out


class TrainLoader(PublicTrainLoader):
    def __init__(self, cfg):
        super(TrainLoader, self).__init__(cfg, resize=True, target_padding=False, to_tensor=True)
        self.transforms.append(UpdateClassIndices())
        self.transforms.append(AssignGTToDefaultBoxes(cfg=cfg))
