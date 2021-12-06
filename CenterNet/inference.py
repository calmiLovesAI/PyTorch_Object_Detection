import torch


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
        self.original_image_size = torch.tensor(original_image_size, dtype=torch.float32, device=self.device)
        self.input_image_size = torch.tensor(input_image_size, dtype=torch.float32, device=self.device)
        self.downsampling_ratio = cfg["Model"]["downsampling_ratio"]
        self.score_threshold = cfg["Decode"]["score_threshold"]

    def __call__(self, outputs):
        heatmap = outputs[..., :self.num_classes]
        reg = outputs[..., self.num_classes: self.num_classes + 2]
        wh = outputs[..., -2:]
        heatmap = torch.sigmoid(heatmap)
        batch_size = heatmap.size()[0]
        heatmap = Decode._nms(heatmap)
        pass

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
        topk_xs = (topk_inds // C % W).to(torch.float32)
        topk_ys = (topk_inds // C // W).to(torch.float32)
        topk_inds = (topk_ys * W + topk_xs).to(torch.int32)
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
