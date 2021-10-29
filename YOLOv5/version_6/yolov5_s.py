import torch
import torch.nn as nn

from YOLOv5.modules import ConvBnAct, C3, SPPF


class YoloV5(nn.Module):
    def __init__(self, num_classes, anchors):
        super(YoloV5, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors[0]) // 2

        self.conv1 = ConvBnAct(3, 32, kernel_size=6, stride=2, padding=2)
        self.conv2 = ConvBnAct(32, 64, 3, 2)
        self.c3_1 = C3(64, 64, 1)
        self.conv3 = ConvBnAct(64, 128, 3, 2)
        self.c3_2 = C3(128, 128, 2)
        self.conv4 = ConvBnAct(128, 256, 3, 2)
        self.c3_3 = C3(256, 256, 3)
        self.conv5 = ConvBnAct(256, 512, 3, 2)
        self.c3_4 = C3(512, 512, 1)
        self.sppf = SPPF(512, 512, 5)
        self.conv6 = ConvBnAct(512, 256, 1, 1)

        self.upsample_1 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.c3_5 = C3(512, 256, 1, False)
        self.conv7 = ConvBnAct(256, 128, 1, 1)
        self.upsample_2 = nn.Upsample(size=None, scale_factor=2, mode="nearest")
        self.c3_6 = C3(256, 128, 1, False)
        self.conv8 = ConvBnAct(128, 128, 3, 2)
        self.c3_7 = C3(256, 256, 1, False)
        self.conv9 = ConvBnAct(256, 256, 3, 2)
        self.c3_8 = C3(512, 512, 1, False)

        self.final_convs = nn.ModuleList(
            nn.Conv2d(c_in, (self.num_classes + 5) * self.num_anchors, kernel_size=(1, 1), stride=(1, 1), padding=0) for
            c_in in [128, 256, 512])

    def _reshape(self, x):
        x = torch.reshape(x, (x.size()[0], self.num_anchors, -1, x.size()[-2], x.size()[-1]))
        x = torch.permute(x, dims=(0, 1, 3, 4, 2))
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        x = self.conv3(x)
        x = self.c3_2(x)
        a1 = x
        x = self.conv4(x)
        x = self.c3_3(x)
        a2 = x
        x = self.conv5(x)
        x = self.c3_4(x)
        x = self.sppf(x)
        x = self.conv6(x)
        a4 = x

        x = self.upsample_1(x)
        x = torch.cat((x, a2), dim=1)
        x = self.c3_5(x)
        x = self.conv7(x)
        a3 = x
        x = self.upsample_2(x)
        x = torch.cat((x, a1), dim=1)
        x = self.c3_6(x)
        o1 = x
        x = self.conv8(x)
        x = torch.cat((x, a3), dim=1)
        x = self.c3_7(x)
        o2 = x
        x = self.conv9(x)
        x = torch.cat((x, a4), dim=1)
        x = self.c3_8(x)
        o3 = x

        o1 = self._reshape(self.final_convs[0](o1))
        o2 = self._reshape(self.final_convs[1](o2))
        o3 = self._reshape(self.final_convs[2](o3))

        return o1, o2, o3
