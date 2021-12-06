import torch

from core.CenterNet.loss import RegL1Loss
from utils.tools import reverse_letter_box


class Decode:
    def __init__(self, cfg, original_image_size, input_image_size):
        """
        初始化参数
        :param cfg:
        :param original_image_size: [h, w]
        :param input_image_size: [H, W]
        """
        self.device = cfg["device"]
        self.K = cfg["Train"]["max_num_boxes"]
        self.num_classes = cfg["Model"]["num_classes"]
        self.original_image_size = original_image_size
        self.input_image_size = input_image_size
        self.downsampling_ratio = cfg["Model"]["downsampling_ratio"]
        self.score_threshold = cfg["Decode"]["score_threshold"]

    def __call__(self, outputs):
        heatmap = outputs[..., :self.num_classes]
        reg = outputs[..., self.num_classes: self.num_classes + 2]
        wh = outputs[..., -2:]
        heatmap = torch.sigmoid(heatmap)
        batch_size = heatmap.size()[0]
        heatmap = Decode._nms(heatmap)
        scores, inds, clses, ys, xs = Decode._top_k(scores=heatmap, k=self.K)
        if reg is not None:
            reg = RegL1Loss.gather_feat(feat=reg, ind=inds)
            xs = torch.reshape(xs, shape=(batch_size, self.K)) + reg[:, :, 0]
            ys = torch.reshape(ys, shape=(batch_size, self.K)) + reg[:, :, 1]
        else:
            xs = torch.reshape(xs, shape=(batch_size, self.K)) + 0.5
            ys = torch.reshape(ys, shape=(batch_size, self.K)) + 0.5
        wh = RegL1Loss.gather_feat(feat=wh, ind=inds)
        clses = torch.reshape(clses, (batch_size, self.K)).to(torch.float32)
        scores = torch.reshape(scores, (batch_size, self.K))
        bboxes = torch.cat(tensors=[xs.unsqueeze(-1) - wh[..., 0:1] / 2,
                                    ys.unsqueeze(-1) - wh[..., 1:2] / 2,
                                    xs.unsqueeze(-1) + wh[..., 0:1] / 2,
                                    ys.unsqueeze(-1) + wh[..., 1:2] / 2], dim=-1)   # shape: (batch_size, self.K, 4)
        bboxes /= (self.input_image_size / self.downsampling_ratio)
        bboxes = reverse_letter_box(h=self.original_image_size[0], w=self.original_image_size[1],
                                    input_size=self.input_image_size, boxes=bboxes)
        bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], min=0, max=self.original_image_size[1] - 1)
        bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], min=0, max=self.original_image_size[0] - 1)

        score_mask = scores >= self.score_threshold   # shape: (batch_size, self.K)

        bboxes = bboxes[score_mask]
        scores = scores[score_mask]
        clses = clses[score_mask]
        return bboxes, scores, clses

    @staticmethod
    def _nms(heatmap, pool_size=3):
        hmax = torch.nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=(pool_size - 1) // 2)(heatmap)
        keep = torch.eq(heatmap, hmax).to(torch.float32)
        return hmax * keep

    @staticmethod
    def _top_k(scores, k):
        B, H, W, C = scores.size()
        scores = torch.reshape(scores, shape=(B, -1))
        topk_scores, topk_inds = torch.topk(input=scores, k=k, largest=True, sorted=True)
        topk_clses = topk_inds % C
        tmp = torch.div(topk_inds, C, rounding_mode="floor")
        topk_xs = (tmp % W).to(torch.float32)
        topk_ys = torch.div(tmp, W, rounding_mode="floor").to(torch.float32)
        topk_inds = (topk_ys * W + topk_xs).to(torch.int32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
