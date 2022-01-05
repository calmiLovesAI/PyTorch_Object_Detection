import torch
import torch.nn.functional as F

from core.SSD.anchor import DefaultBoxes


class MultiBoxLoss:
    def __init__(self, cfg):
        self.device = cfg["device"]
        # torch.Tensor, shape: (先验框总数(8732), 4)
        self.default_boxes = DefaultBoxes(cfg).__call__(xyxy=False).to(self.device)
        self.default_boxes.requires_grad = False
        self.default_boxes = self.default_boxes.unsqueeze(dim=0)  # shape: (1, 8732, 4)
        self.num_classes = cfg["Model"]["num_classes"] + 1
        self.threshold = cfg["Loss"]["overlap_thresh"]
        self.variance = cfg["Loss"]["variance"]
        self.negpos_ratio = cfg["Loss"]["neg_pos"]
        self.scale_xy = 1.0 / self.variance[0]
        self.scale_wh = 1.0 / self.variance[1]

    def _location_vec(self, loc):
        g_cxcy = self.scale_xy * (loc[..., :2] - self.default_boxes[..., :2]) / self.default_boxes[..., 2:]
        g_wh = self.scale_wh * torch.log(loc[..., 2:] / self.default_boxes[..., 2:])
        return torch.cat(tensors=(g_cxcy, g_wh), dim=-1)

    def __call__(self, y_true, y_pred):
        """

        Args:
            y_true: torch.Tensor, shape: (batch_size, 8732, 5(cx, cy, w, h, class_index))
            y_pred: (loc, conf), 其中loc的shape是(batch_size, 8732, 4), conf的shape是(batch_size, 8732, self.num_classes)

        Returns:

        """
        ploc, plabel = y_pred
        gloc = y_true[..., :-1]  # (batch_size, 8732, 4)
        glabel = y_true[..., -1].long()  # (batch_size, 8732)

        # 筛选正样本
        mask = glabel > 0  # (batch_size, 8732)
        # 正样本个数
        pos_num = mask.sum(dim=1)  # (batch_size)

        # 偏移量
        vec_gd = self._location_vec(gloc)  # (batch_size, 8732, 4)
        # 位置损失
        loc_loss = F.smooth_l1_loss(ploc, vec_gd, reduction="none").sum(dim=-1)  # (batch_size, 8732)
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # (batch_size)

        con = F.cross_entropy(torch.permute(plabel, dims=(0, 2, 1)), glabel, reduction="none")  # (batch_size, 8732)

        # Hard Negative Mining
        con_neg = con.clone()
        con_neg[mask] = torch.tensor(0.0)
        # 排序，得到一个索引，它的值表示这个位置的元素第几大
        _, con_idx = con_neg.sort(1, descending=True)
        _, con_rank = con_idx.sort(1)
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(1)  # (batch_size, 1)
        neg_mask = con_rank < neg_num  # (batch_size, 8732)

        # 分类损失
        con_loss = (con * (mask.float() + neg_mask.float())).sum(1)  # (batch_size)

        total_loss = loc_loss + con_loss
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        total_loss = (total_loss * num_mask / pos_num).mean(dim=0)
        loss_l = (loc_loss * num_mask / pos_num).mean(dim=0)
        loss_c = (con_loss * num_mask / pos_num).mean(dim=0)
        return total_loss, loss_l, loss_c


"""
The following code comes from: https://github.com/amdegroot/ssd.pytorch
"""


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

