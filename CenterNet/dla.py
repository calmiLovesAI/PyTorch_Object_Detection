import torch
import torch.nn as nn
import torch.nn.functional as F


def same_pad(k, s):
    return (k - s) / 2


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(stride, stride),
                               padding=same_pad(3, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=c_out)
        self.conv2 = nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1),
                               padding=same_pad(3, stride), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=c_out)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class BottleNeck(nn.Module):
    expansion = 2

    def __init__(self, c_in, c_out, stride=1):
        super(BottleNeck, self).__init__()
        c_t = c_out // BottleNeck.expansion
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_t, kernel_size=(1, 1), stride=(1, 1),
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=c_t)
        self.conv2 = nn.Conv2d(in_channels=c_t, out_channels=c_t, kernel_size=(3, 3), stride=(stride, stride),
                               padding=same_pad(3, stride), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=c_t)
        self.conv3 = nn.Conv2d(in_channels=c_t, out_channels=c_out, kernel_size=(1, 1), stride=(1, 1),
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=c_out)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + residual)
        return x


class BottleNeckX(nn.Module):
    cardinality = 32

    def __init__(self, c_in, c_out, stride=1):
        super(BottleNeckX, self).__init__()
        c_t = c_in * BottleNeckX.cardinality // 32
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_t, kernel_size=(1, 1), stride=(1, 1),
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=c_t)
        self.conv2 = nn.Conv2d(in_channels=c_t, out_channels=c_t, kernel_size=(3, 3), stride=(stride, stride),
                               padding=same_pad(3, stride), groups=BottleNeckX.cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=c_t)
        self.conv3 = nn.Conv2d(in_channels=c_t, out_channels=c_out, kernel_size=(1, 1), stride=(1, 1),
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=c_out)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + residual)
        return x
