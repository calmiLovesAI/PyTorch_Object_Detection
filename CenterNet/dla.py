import torch
import torch.nn as nn
import torch.nn.functional as F


def same_pad(k, s):
    return (k - s) // 2


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(stride, stride),
                               padding=same_pad(3, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=c_out)
        self.conv2 = nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1),
                               padding=1, bias=False)
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


class Root(nn.Module):
    def __init__(self, c_in, c_out, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.bn = nn.BatchNorm2d(num_features=c_out)
        self.residual = residual

    def forward(self, inputs):
        x = self.bn(self.conv(torch.cat(inputs, dim=1)))
        if self.residual:
            x += inputs[0]
        x = F.relu(x)
        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, 1)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels

        if self.levels == 1:
            self.root = Root(root_dim, out_channels, root_residual)

        if stride > 1:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, inputs, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(inputs) if self.downsample else inputs
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(inputs, residual=residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            outputs = self.root([x2, x1, *children])
        else:
            children.append(x1)
            outputs = self.tree2(x1, children=children)
        return outputs


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock,
                 residual_root=False, return_levels=False, pool_size=7):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes

        self.base_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False),
            nn.BatchNorm2d(num_features=channels[0]),
            nn.ReLU()
        )

        self.level_0 = DLA._make_conv_level(channels[0], channels[0], levels[0])
        self.level_1 = DLA._make_conv_level(channels[0], channels[1], levels[1], 2)
        self.level_2 = Tree(levels=levels[2], block=block, in_channels=channels[1],
                            out_channels=channels[2], stride=2,
                            level_root=False, root_residual=residual_root)
        self.level_3 = Tree(levels=levels[3], block=block, in_channels=channels[2],
                            out_channels=channels[3], stride=2,
                            level_root=True, root_residual=residual_root)
        self.level_4 = Tree(levels=levels[4], block=block, in_channels=channels[3],
                            out_channels=channels[4], stride=2,
                            level_root=True, root_residual=residual_root)
        self.level_5 = Tree(levels=levels[5], block=block, in_channels=channels[4],
                            out_channels=channels[5], stride=2,
                            level_root=True, root_residual=residual_root)
        self.avgpool = nn.AvgPool2d(kernel_size=pool_size)
        self.final = nn.Conv2d(in_channels=channels[5], out_channels=num_classes, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True)


    @staticmethod
    def _make_conv_level(in_channels, out_channels, convs, stride=1):
        layers = []
        for i in range(convs):
            if i == 0:
                layers.extend([
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=(stride, stride),
                              padding=same_pad(3, stride),
                              bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU()
                ])
            else:
                layers.extend([
                    nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              stride=(1, 1),
                              padding=1,
                              bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                    nn.ReLU()
                ])
        return nn.Sequential(*layers)

    def forward(self, x):
        y = list()
        x = self.base_layer(x)
        x = self.level_0(x)
        y.append(x.clone())
        x = self.level_1(x)
        y.append(x.clone())
        x = self.level_2(x)
        y.append(x.clone())
        x = self.level_3(x)
        y.append(x.clone())
        x = self.level_4(x)
        y.append(x.clone())
        x = self.level_5(x)
        y.append(x.clone())

        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.final(x)
            x = torch.reshape(x, (x.size()[0], -1))
            return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class IDAUp(nn.Module):
    def __init__(self, in_dim, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(in_channels=in_dim,
                              out_channels=out_dim,
                              kernel_size=(1, 1),
                              stride=(1, 1),
                              padding=0,
                              bias=False),
                    nn.BatchNorm2d(num_features=out_dim),
                    nn.ReLU()
                )
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(in_channels=out_dim, out_channels=out_dim, kernel_size=f*2, stride=f, padding=f, output_padding=f, groups=out_dim, bias=False)
            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(in_channels=(f+1)*out_dim, out_channels=out_dim, kernel_size=node_kernel, strides=1, padding=same_pad(node_kernel, 1), bias=False),
                nn.BatchNorm2d(num_features=out_dim),
                nn.ReLU()
            )
            setattr(self, "node_" + str(i), node)

    def forward(self, inputs):
        pass


# if __name__ == '__main__':
#     sample = torch.randn(1, 3, 384, 384, dtype=torch.float32)
#     model = DLA(levels=[1, 1, 1, 2, 2, 1], channels=[16, 32, 64, 128, 256, 512], block=BasicBlock,
#                        return_levels=True)
#     y = model(sample)
#     for f in y:
#         print(f.size())
    # torch.Size([1, 16, 384, 384])
    # torch.Size([1, 32, 191, 191])
    # torch.Size([1, 64, 95, 95])
    # torch.Size([1, 128, 47, 47])
    # torch.Size([1, 256, 23, 23])
    # torch.Size([1, 512, 11, 11])