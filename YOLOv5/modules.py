import torch
import torch.nn as nn
import torch.functional as F


def same_pad(k, s):
    return (k - s) / 2


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=same_pad(kernel_size, stride) if padding is None else padding,
                              groups=groups)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride):
        super(Focus, self).__init__()
        # 分组卷积
        self.conv = ConvBnAct(c_in * 4, c_out, kernel_size, stride, padding=None, groups=4)

    def forward(self, x):
        x = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    """
    残差网络结构
    """

    def __init__(self, c_in, c_out, shortcut=True, g=1):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBnAct(c_in, c_out / 2, 1, 1)
        self.conv2 = ConvBnAct(c_out / 2, c_out, 3, 1, groups=g)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self):
        super(BottleneckCSP, self).__init__()
        pass

    def forward(self, x):
        pass


class C3(nn.Module):
    """
    使用了3个卷积的CSP模块
    """
    def __init__(self, c_in, c_out, n=1, shortcut=True, g=1):
        super(C3, self).__init__()
        c_ = int(c_out / 2)  # 中间层的通道数
        self.conv1 = ConvBnAct(c_in, c_, 1, 1)
        self.conv2 = ConvBnAct(c_in, c_, 1, 1)
        self.conv3 = ConvBnAct(2 * c_, c_out, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g) for _ in range(n)])

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.m(x1)
        x2 = self.conv2(x)
        x3 = torch.cat((x1, x2), dim=1)
        return self.conv3(x3)


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling
    """

    def __init__(self, c_in, c_out, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c_in // 2
        self.conv1 = ConvBnAct(c_in, c_, 1, 1)
        self.conv2 = ConvBnAct(c_ * (len(k) + 1), c_out, 1, 1)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.pools], dim=1)
        x = self.conv2(x)
        return x


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling Fast
    """

    def __init__(self, c_in, c_out, k=5):
        super(SPPF, self).__init__()
        c_ = c_in // 2
        self.conv1 = ConvBnAct(c_in, c_, 1, 1)
        self.conv2 = ConvBnAct(c_ * 4, c_out, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.pool(x)
        x2 = self.pool(x1)
        x3 = self.pool(x2)
        return self.conv2(torch.cat((x, x1, x2, x3), dim=1))

