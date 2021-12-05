import torch
import torch.nn as nn

from CenterNet.dla import DLASeg


class CenterNet(nn.Module):
    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        self.heads = {"heatmap": cfg["Model"]["num_classes"], "wh": 2, "reg": 2}
        self.backbone = DLASeg(base_name="dla34", heads=self.heads)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.cat(tensors=x, dim=1)
        x = torch.permute(x, dims=(0, 2, 3, 1))
        return x
