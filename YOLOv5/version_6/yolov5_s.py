import torch
import torch.nn as nn

from YOLOv5.modules import ConvBnAct, C3, SPPF


class YoloV5(nn.Module):
    def __init__(self):
        super(YoloV5, self).__init__()
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        x = self.conv3(x)
        x = self.c3_2(x)
        x = self.conv4(x)
        x = self.c3_3(x)
        x = self.conv5(x)
        x = self.c3_4(x)
        x = self.sppf(x)
        x = self.conv6(x)
