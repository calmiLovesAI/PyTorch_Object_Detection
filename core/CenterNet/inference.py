import torch

from core.CenterNet.loss import RegL1Loss
from utils.heatmap import visualize_heatmap
from utils.tools import reverse_letter_box


class Decode:
    def __init__(self, cfg, original_image_size, input_image_size):
        """
        初始化参数
        :param cfg:
        :param original_image_size: [h, w]
        :param input_image_size: int
        """
        self.device = cfg["device"]
        self.K = cfg["Decode"]["max_boxes_per_img"]
        self.num_classes = cfg["Model"]["num_classes"]
        self.original_image_size = original_image_size
        self.input_image_size = input_image_size
        self.downsampling_ratio = cfg["Model"]["downsampling_ratio"]
        self.feature_size = self.input_image_size / self.downsampling_ratio
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
            xs = torch.reshape(xs, shape=(batch_size, self.K)) + reg[:, :, 0]   # shape: (batch_size, self.K)
            ys = torch.reshape(ys, shape=(batch_size, self.K)) + reg[:, :, 1]
        else:
            xs = torch.reshape(xs, shape=(batch_size, self.K)) + 0.5
            ys = torch.reshape(ys, shape=(batch_size, self.K)) + 0.5
        wh = RegL1Loss.gather_feat(feat=wh, ind=inds)    # shape: (batch_size, self.K, 2)
        clses = torch.reshape(clses, (batch_size, self.K))
        scores = torch.reshape(scores, (batch_size, self.K))

        bboxes = torch.cat(tensors=[xs.unsqueeze(-1), ys.unsqueeze(-1), wh], dim=-1)  # shape: (batch_size, self.K, 4)

        bboxes /= self.feature_size
        bboxes = torch.clamp(bboxes, min=0, max=1)
        bboxes = reverse_letter_box(h=self.original_image_size[0], w=self.original_image_size[1],
                                    input_size=self.input_image_size, boxes=bboxes)

        score_mask = scores >= self.score_threshold   # shape: (batch_size, self.K)

        bboxes = bboxes[score_mask]
        scores = scores[score_mask]
        clses = clses[score_mask]
        return bboxes, scores, clses

    @staticmethod
    def _nms(heatmap, pool_size=3):
        hmax = torch.nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=((pool_size - 1) // 2))(heatmap)
        keep = torch.eq(heatmap, hmax).to(torch.float32)
        return heatmap * keep

    @staticmethod
    def _top_k(scores, k):
        B, H, W, C = scores.size()
        scores = torch.reshape(scores, shape=(B, -1))
        topk_scores, topk_inds = torch.topk(input=scores, k=k, largest=True, sorted=True)
        topk_clses = topk_inds % C   # 应该选取哪些通道（类别）
        pixel = torch.div(topk_inds, C, rounding_mode="floor")
        topk_ys = torch.div(pixel, W, rounding_mode="floor")    # 中心点的y坐标
        topk_xs = pixel % W    # 中心点的x坐标
        topk_inds = (topk_ys * W + topk_xs).to(torch.int32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
