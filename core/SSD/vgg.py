import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, use_bn=False):
        super(VGG, self).__init__()
        self.conv1 = VGG._make_conv_block(3, 64, 3, 1, 1, use_bn)
        self.conv2 = VGG._make_conv_block(64, 64, 3, 1, 1, use_bn)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = VGG._make_conv_block(64, 128, 3, 1, 1, use_bn)
        self.conv4 = VGG._make_conv_block(128, 128, 3, 1, 1, use_bn)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv5 = VGG._make_conv_block(128, 256, 3, 1, 1, use_bn)
        self.conv6 = VGG._make_conv_block(256, 256, 3, 1, 1, use_bn)
        self.conv7 = VGG._make_conv_block(256, 256, 3, 1, 1, use_bn)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = VGG._make_conv_block(256, 512, 3, 1, 1, use_bn)
        self.conv9 = VGG._make_conv_block(512, 512, 3, 1, 1, use_bn)
        self.conv10 = VGG._make_conv_block(512, 512, 3, 1, 1, use_bn)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = VGG._make_conv_block(512, 512, 3, 1, 1, use_bn)
        self.conv12 = VGG._make_conv_block(512, 512, 3, 1, 1, use_bn)
        self.conv13 = VGG._make_conv_block(512, 512, 3, 1, 1, use_bn)

        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv15 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1)


    @staticmethod
    def _make_conv_block(c_in, c_out, k, s, p, bn=True):
        if bn:
            return nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p),
                nn.BatchNorm2d(num_features=c_out),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p),
                nn.ReLU()
            )

    def forward(self, x):
        """

        Args:
            x: torch.Tensor, shape: (N, C, 300, 300)

        Returns: list of torch.Tensor, shape: [(N, 512, 38, 38), (N, 1024, 19, 19)]

        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        o1 = x
        x = self.pool4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.pool5(x)
        x = self.conv14(x)
        x = F.relu(x)
        x = self.conv15(x)
        x = F.relu(x)
        o2 = x

        return o1, o2
