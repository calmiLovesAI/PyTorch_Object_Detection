import torch
import torch.nn as nn
import torch.functional as F


def same_pad(k, s):
    return (k - s) / 2


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, groups=1):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=same_pad(kernel_size, stride) if padding is None else padding,
                              groups=groups)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Focus, self).__init__()
        # 分组卷积
        self.conv = CBL(in_channels * 4, out_channels, kernel_size, stride, padding=None, groups=4)

    def forward(self, x):
        x = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    """
    残差网络结构
    """
    def __init__(self):
        super(Bottleneck, self).__init__()
        pass

    def forward(self, x):
        pass


class CSPWithBottleneck(nn.Module):
    """
    第一种CSP结构，使用了残差模块
    """
    def __init__(self):
        super(CSPWithBottleneck, self).__init__()
        pass

    def forward(self, x):
        pass


class CSPWithConv(nn.Module):
    """
    第二种CSP结构，使用卷积模块替代了残差模块
    """
    def __init__(self):
        super(CSPWithConv, self).__init__()
        pass

    def forward(self, x):
        pass

