import torch
import torch.nn as nn

from core.SSD.vgg import VGG
from utils.auto_padding import same_padding


class L2Normalize(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Normalize, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class ExtraLayer(nn.Module):
    def __init__(self, c_in):
        super(ExtraLayer, self).__init__()
        self.conv1 = ExtraLayer._make_layer(c_in, 256, 1, 1)
        self.conv2 = ExtraLayer._make_layer(256, 512, 3, 2)
        self.conv3 = ExtraLayer._make_layer(512, 128, 1, 1)
        self.conv4 = ExtraLayer._make_layer(128, 256, 3, 2)
        self.conv5 = ExtraLayer._make_layer(256, 128, 1, 1)
        self.conv6 = ExtraLayer._make_layer(128, 256, 3, 1, False)
        self.conv7 = ExtraLayer._make_layer(256, 128, 1, 1)
        self.conv8 = ExtraLayer._make_layer(128, 256, 3, 1, False)

    @staticmethod
    def _make_layer(c_in, c_out, k, s, same=True):
        if same:
            return nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=same_padding(k, s)),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=0),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        o1 = x
        x = self.conv3(x)
        x = self.conv4(x)
        o2 = x
        x = self.conv5(x)
        x = self.conv6(x)
        o3 = x
        x = self.conv7(x)
        x = self.conv8(x)
        o4 = x

        return o1, o2, o3, o4


class SSD(nn.Module):
    def __init__(self, cfg):
        super(SSD, self).__init__()
        self.num_classes = cfg["Model"]["num_classes"] + 1
        # self.stage_boxes_per_pixel: 每个stage分支输出的feature map中每个像素位置处的先验框数量
        aspect_ratio = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.stage_boxes_per_pixel = [2*len(ar) + 2 for ar in aspect_ratio]

        self.backbone = VGG(use_bn=True)
        self.l2_norm = L2Normalize(n_channels=512, scale=20)
        self.extras = ExtraLayer(c_in=1024)
        self.locs, self.confs = self._make_locs_and_confs(num_classes=self.num_classes)

    def _make_locs_and_confs(self, num_classes):
        loc_layers = nn.ModuleList()
        conf_layers = nn.ModuleList()
        channels = [512, 1024, 512, 256, 256, 256]
        for i in range(len(channels)):
            loc_layers.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=self.stage_boxes_per_pixel[i] * 4,
                    kernel_size=3,
                    stride=1,
                    padding=same_padding(3, 1)
                ))
            conf_layers.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=self.stage_boxes_per_pixel[i] * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=same_padding(3, 1)
                )
            )
        return loc_layers, conf_layers

    def forward(self, x):
        """

        Args:
            x: torch.Tensor, shape: (N, C, H, W)

        Returns: torch.Size([N, 8732, 4]) torch.Size([N, 8732, num_classes + 1])

        """
        sources = list()
        loc = list()
        conf = list()

        x1, x = self.backbone(x)
        x1 = self.l2_norm(x1)
        sources.append(x1)
        sources.append(x)
        o1, o2, o3, o4 = self.extras(x)
        sources.extend([o1, o2, o3, o4])

        for (x, l, c) in zip(sources, self.locs, self.confs):
            loc.append(l(x))
            conf.append(c(x))

        loc = torch.cat(tensors=[torch.reshape(o, shape=(o.size()[0], -1)) for o in loc], dim=1)
        conf = torch.cat(tensors=[torch.reshape(o, shape=(o.size()[0], -1)) for o in conf], dim=1)

        loc = torch.reshape(loc, shape=(loc.shape[0], -1, 4))
        conf = torch.reshape(conf, shape=(conf.shape[0], -1, self.num_classes))

        return loc, conf
