import torch
import torch.nn as nn

from utils.auto_padding import same_padding


class DeformableConv2d(nn.Module):
    """
    可变性卷积，Ref: https://github.com/dontLoveBugs/Deformable_ConvNet_pytorch/blob/master/network/deform_conv/deform_conv_v2.py
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, modulation=False):
        super(DeformableConv2d, self).__init__()

        self.k = kernel_size
        pad = same_padding(kernel_size, stride)
        self.zero_padding = nn.ZeroPad2d(padding=pad)
        self.s = stride
        self.modulation = modulation

        if self.modulation:
            # 卷积核的每个位置都有不同的权重
            self.m_conv = nn.Conv2d(in_channels=in_channels, out_channels=kernel_size * kernel_size,
                                    kernel_size=kernel_size,
                                    stride=stride, padding=pad)

        # 为卷积核的每个位置都生成横纵坐标的偏移量(offset)
        self.p_conv = nn.Conv2d(in_channels=in_channels, out_channels=2 * kernel_size * kernel_size,
                                kernel_size=kernel_size,
                                stride=stride, padding=pad)

        # 最终实际要进行的卷积操作，注意这里步长设置为卷积核大小，
        # 因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=kernel_size, bias=bias)

    def forward(self, x):
        """
        x : torch.Tensor, shape: (B, C, H, W)
        """
        offset = self.p_conv(x)  # (B, 2k^2, H, W)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))  # (B, k^2, H, W)

        data_type = offset.data.type()
        N = self.k * self.k
        x = self.zero_padding(x)

        p = self._get_p(offset, data_type)  # shape: (batch_size, 2N, out_h, out_w)
        p = p.contiguous().permute(0, 2, 3, 1)   # shape: (batch_size, out_h, out_w, 2N)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        # 左上，shape: (batch_size, out_h, out_w, 2N)
        q_lt = torch.cat(tensors=[
            torch.clamp(input=q_lt[..., :N], min=0, max=x.size(2)-1),
            torch.clamp(input=q_lt[..., N:], min=0, max=x.size(3)-1)
        ], dim=-1).long()
        # 右下，shape: (batch_size, out_h, out_w, 2N)
        q_rb = torch.cat(tensors=[
            torch.clamp(input=q_rb[..., :N], min=0, max=x.size(2) - 1),
            torch.clamp(input=q_rb[..., N:], min=0, max=x.size(3)-1)
        ], dim=-1).long()
        # 左下，shape: (batch_size, out_h, out_w, 2N)
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        # 右上，shape: (batch_size, out_h, out_w, 2N)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, self.k)   # (b, c, h * k, w * k)
        out = self.conv(x_offset)

        return out


    def _init_weights(self):
        # 初始化offset为0
        nn.init.constant_(self.p_conv.weight, 0)

        if self.modulation:
            # 初始化所有kernel position权重为1
            nn.init.constant_(self.m_conv.weight, 1)

    def _get_p(self, offset, data_type):
        h, w = offset.shape[2:]
        p0 = self._get_p0(h, w, data_type)
        pn = self._get_pn(data_type)
        p = p0 + pn + offset
        return p

    def _get_p0(self, out_h, out_w, data_type):
        # 将输出feature map上的每一个点匹配到输入feature map的卷积核中心位置
        N = self.k * self.k
        # 卷积核的中心位置
        kc = self.k // 2
        # p0_y shape: (out_h, out_w), p0_x shape: (out_h, out_w)
        p0_y, p0_x = torch.meshgrid(
            torch.arange(start=kc, end=out_h * self.s + kc, step=self.s),
            torch.arange(start=kc, end=out_w * self.s + kc, step=self.s)
        )
        # shape: (1, N, out_h, out_w)
        p0_y = p0_y.flatten().view(1, 1, out_h, out_w).repeat(1, N, 1, 1)
        p0_x = p0_x.flatten().view(1, 1, out_h, out_w).repeat(1, N, 1, 1)
        # (1, 2N, out_h, out_w)
        p0 = torch.cat(tensors=[p0_y, p0_x], dim=1).type(data_type)

        return p0

    def _get_pn(self, data_type):
        # 卷积核的每个位置相对于其中心的偏移(offset)
        N = self.k * self.k
        # shape: (k, k), (k, k)
        pn_y, pn_x = torch.meshgrid(
            torch.arange(start=-(self.k//2), end=self.k//2+1, step=1),
            torch.arange(start=-(self.k//2), end=self.k//2+1, step=1)
        )
        pn_y = pn_y.flatten().view(1, N, 1, 1)
        pn_x = pn_x.flatten().view(1, N, 1, 1)
        # (1, 2N, 1, 1)
        pn = torch.cat(tensors=[pn_y, pn_x], dim=1).type(data_type)

        return pn

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


