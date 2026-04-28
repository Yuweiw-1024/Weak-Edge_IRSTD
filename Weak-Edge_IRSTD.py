import torch
import torch.nn as nn
import torch.nn.functional as F
class ChannelAttention_max(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_max, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride,
                padding=padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        return self.sigmoid(w)


class CDCConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta=0.7):
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out_normal = self.conv(x)
        if abs(self.theta) > 1e-8:
            weight = self.conv.weight
            kernel_diff = weight.sum(dim=(2, 3), keepdim=True)
            out_diff = F.conv2d(
                x, kernel_diff, bias=None,
                stride=self.conv.stride,
                padding=0,
                groups=self.conv.groups,
            )
            out = out_normal - self.theta * out_diff
        else:
            out = out_normal
        out = self.bn(out)
        out = self.act(out)
        return out


def _split_channels(total_channels, num_splits=3):
    base = total_channels // num_splits
    rem = total_channels % num_splits
    return [base + (1 if i < rem else 0) for i in range(num_splits)]


class DWPWBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.dw = ConvBNReLU(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            groups=in_channels
        )
        self.pw = ConvBNReLU(
            in_channels, out_channels,
            kernel_size=1, padding=0, stride=1, groups=1
        )

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class ChannelResidualMLP(nn.Module):
    """
    返回“已经加过残差”的结果。
    外部不要再写 x + module(x)
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        delta = self.pool(x)
        delta = self.fc1(delta)
        delta = self.act(delta)
        delta = self.fc2(delta)
        return x + delta


class BiDirectionalFusion(nn.Module):
    """
    双向、非对称深浅融合块

    Md: 深层特征（已上采样到与 Ms 同分辨率）
    Ms: 浅层特征

    输出:
        torch.cat([ms_out, md_out], dim=1)

    参数:
        md_channels: 深层输入/输出通道
        ms_in_channels: 浅层输入通道
        ms_out_channels: 浅层输出通道
            - 若为 None，则默认与 ms_in_channels 相同
            - 用于处理 0_4 这种 Ms 输入 32ch，但最终只想输出 16ch 的情况
    """
    def __init__(self, md_channels, ms_in_channels, ms_out_channels=None, reduction=4, cdc_theta=0.7):
        super().__init__()
        if ms_out_channels is None:
            ms_out_channels = ms_in_channels

        self.md_channels = md_channels
        self.ms_in_channels = ms_in_channels
        self.ms_out_channels = ms_out_channels

        # ===== 1) 深层语义整理分支 =====
        self.md_se = SEBlock(md_channels, reduction=reduction)
        md_splits = _split_channels(md_channels, 3)
        self.md_branches = nn.ModuleList([
            DWPWBranch(md_channels, md_splits[0], kernel_size=3),
            DWPWBranch(md_channels, md_splits[1], kernel_size=5),
            DWPWBranch(md_channels, md_splits[2], kernel_size=7),
        ])
        self.md_fuse = ConvBNReLU(sum(md_splits), md_channels, kernel_size=1, padding=0)

        # ===== 2) 浅层细节净化分支 =====
        self.ms_cdc = CDCConv2d(ms_in_channels, ms_out_channels, kernel_size=3, stride=1, padding=1, theta=cdc_theta)
        self.ms_branch_1 = nn.Sequential(
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1),
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.ms_branch_2 = nn.Sequential(
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1),
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.ms_branch_3 = ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1)
        self.ms_detail_fuse = ConvBNReLU(ms_out_channels * 3, ms_out_channels, kernel_size=1, padding=0)
        self.ms_detail_refine = ChannelResidualMLP(ms_out_channels, reduction=reduction)

        # 原始浅层 skip 投影到 ms_out_channels，避免浅层信息被完全改写
        self.ms_skip_proj = ConvBNReLU(ms_in_channels, ms_out_channels, kernel_size=1, padding=0)

        # ===== 3) 深 -> 浅：成熟语义指导浅层 =====
        self.md_to_ms = ConvBNReLU(md_channels, ms_out_channels, kernel_size=1, padding=0)
        self.md_gate = nn.Sequential(
            nn.Conv2d(ms_out_channels, ms_out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.md_bias = ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=1, padding=0)

        # 浅层最终融合
        self.ms_out_fuse = ConvBNReLU(ms_out_channels * 3, ms_out_channels, kernel_size=1, padding=0)

        # ===== 4) 浅 -> 深：干净细节补偿深层 =====
        self.ms_to_md = ConvBNReLU(ms_out_channels, md_channels, kernel_size=1, padding=0)

        self.ms_bias = ConvBNReLU(md_channels, md_channels, kernel_size=1, padding=0)

        self.avgpool = nn.AvgPool2d((3, 3), stride=1, padding=1)
    def forward(self, Md, Ms):
        # -------------------------------------------------
        # A. 深层：先形成成熟语义 md_feat
        # -------------------------------------------------
        md_att = self.md_se(Md) * Md
        md_multi = [branch(md_att) for branch in self.md_branches]
        md_feat = self.md_fuse(torch.cat(md_multi, dim=1))

        # -------------------------------------------------
        # B. 浅层：先形成干净细节 ms_clean
        # -------------------------------------------------

        Ms = Ms - self.avgpool(Ms) + Ms

        # ms_base = self.ms_cdc(Ms)
        # ms_1 = self.ms_branch_1(ms_base)
        # ms_2 = self.ms_branch_2(ms_base)
        # ms_3 = self.ms_branch_3(ms_base)
        # ms_detail = self.ms_detail_fuse(torch.cat([ms_1, ms_2, ms_3], dim=1))
        # ms_clean = self.ms_detail_refine(ms_detail)   # 注意：这里不要外面再 + 一次
        # ms_skip = self.ms_skip_proj(Ms)

        # -------------------------------------------------
        # C. 深 -> 浅：用成熟语义 md_feat 指导浅层
        # -------------------------------------------------
        # md_proj = self.md_to_ms(md_feat)
        # md_gate = self.md_gate(md_proj)
        # md_bias = self.md_bias(md_proj)

        # 非对称指导：语义对细节做“门控 + 偏置”
        # ms_guided = ms_clean * (1.0 + md_gate) + md_bias
        # ms_guided = ms_clean * md_gate + md_bias
        # ms_out = self.ms_out_fuse(torch.cat([ms_guided, ms_skip, md_proj], dim=1))

        # -------------------------------------------------
        # D. 浅 -> 深：用干净细节 ms_clean 补偿深层
        # -------------------------------------------------

        # ms_proj = self.ms_to_md(ms_clean)
        # ms_bias = self.ms_bias(ms_proj)
        # md_out = md_feat + ms_bias
        # md_out = md_feat + ms_bias



        # 保持原 decoder 的 cat 顺序：[浅层, 深层]
        return torch.cat([Ms, md_feat], dim=1)

class BiDirectionalFusion_qian(nn.Module):

    def __init__(self, md_channels, ms_in_channels, ms_out_channels=None, reduction=4, cdc_theta=0.7):
        super().__init__()
        if ms_out_channels is None:
            ms_out_channels = ms_in_channels

        self.md_channels = md_channels
        self.ms_in_channels = ms_in_channels
        self.ms_out_channels = ms_out_channels

        # ===== 1) 深层语义整理分支 =====
        self.md_se = SEBlock(md_channels, reduction=reduction)
        md_splits = _split_channels(md_channels, 3)
        self.md_branches = nn.ModuleList([
            DWPWBranch(md_channels, md_splits[0], kernel_size=3),
            DWPWBranch(md_channels, md_splits[1], kernel_size=5),
            DWPWBranch(md_channels, md_splits[2], kernel_size=7),
        ])
        self.md_fuse = ConvBNReLU(sum(md_splits), md_channels, kernel_size=1, padding=0)

        # ===== 2) 浅层细节净化分支 =====
        self.ms_cdc = CDCConv2d(ms_in_channels, ms_out_channels, kernel_size=3, stride=1, padding=1, theta=cdc_theta)
        self.ms_branch_1 = nn.Sequential(
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1),
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.ms_branch_2 = nn.Sequential(
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1),
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.ms_branch_3 = ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1)
        self.ms_detail_fuse = ConvBNReLU(ms_out_channels * 3, ms_out_channels, kernel_size=1, padding=0)
        self.ms_detail_refine = ChannelResidualMLP(ms_out_channels, reduction=reduction)

        # 原始浅层 skip 投影到 ms_out_channels，避免浅层信息被完全改写
        self.ms_skip_proj = ConvBNReLU(ms_in_channels, ms_out_channels, kernel_size=1, padding=0)

        # ===== 3) 深 -> 浅：成熟语义指导浅层 =====
        self.md_to_ms = ConvBNReLU(md_channels, ms_out_channels, kernel_size=1, padding=0)
        self.md_gate = nn.Sequential(
            nn.Conv2d(ms_out_channels, ms_out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.md_bias = ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=1, padding=0)

        # 浅层最终融合
        self.ms_out_fuse = ConvBNReLU(ms_out_channels * 3, ms_out_channels, kernel_size=1, padding=0)

        # ===== 4) 浅 -> 深：干净细节补偿深层 =====
        self.ms_to_md = ConvBNReLU(ms_out_channels, md_channels, kernel_size=3, stride=1, padding=1)

        self.ms_bias = ConvBNReLU(md_channels, md_channels, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
    def forward(self, Md, Ms):
        ms_out = Ms
        # -------------------------------------------------
        # A. 深层：先形成成熟语义 md_feat
        # -------------------------------------------------
        md_att = self.md_se(Md) * Md
        md_multi = [branch(md_att) for branch in self.md_branches]
        md_feat = self.md_fuse(torch.cat(md_multi, dim=1))

        # -------------------------------------------------
        # D. 浅 -> 深：用干净细节 ms_clean 补偿深层
        # -------------------------------------------------
        ms_proj = self.ms_to_md(Ms)
        ms_bias = self.ms_bias(ms_proj)
        md_out = self.relu(md_feat + ms_bias)

        # 保持原 decoder 的 cat 顺序：[浅层, 深层]
        return torch.cat([ms_out, md_out], dim=1)


class BiDirectionalFusion_enc(nn.Module):

    def __init__(self, md_channels, ms_in_channels, ms_out_channels=None, reduction=4, cdc_theta=0.7):
        super().__init__()
        if ms_out_channels is None:
            ms_out_channels = ms_in_channels

        self.md_channels = md_channels
        self.ms_in_channels = ms_in_channels
        self.ms_out_channels = ms_out_channels

        # ===== 2) 浅层细节净化分支 =====
        self.ms_cdc = CDCConv2d(ms_in_channels, ms_out_channels, kernel_size=3, stride=1, padding=1, theta=cdc_theta)
        self.ms_branch_1 = nn.Sequential(
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1),
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.ms_branch_2 = nn.Sequential(
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1),
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.ms_branch_3 = ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1)
        self.ms_detail_fuse = ConvBNReLU(ms_out_channels * 3, ms_out_channels, kernel_size=1, padding=0)
        self.ms_detail_refine = ChannelResidualMLP(ms_out_channels, reduction=reduction)

        # 原始浅层 skip 投影到 ms_out_channels，避免浅层信息被完全改写
        self.ms_skip_proj = ConvBNReLU(ms_in_channels, ms_out_channels, kernel_size=1, padding=0)

        # ===== 3) 深 -> 浅：成熟语义指导浅层 =====
        self.md_to_ms = ConvBNReLU(md_channels, ms_out_channels, kernel_size=1, padding=0)
        self.md_gate = nn.Sequential(
            nn.Conv2d(ms_out_channels, ms_out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.md_bias = ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=1, padding=0)

        # 浅层最终融合
        self.ms_out_fuse = ConvBNReLU(ms_out_channels * 3, ms_out_channels, kernel_size=1, padding=0)

        # ===== 4) 浅 -> 深：干净细节补偿深层 =====
        self.ms_to_md = ConvBNReLU(md_channels, md_channels, kernel_size=3, stride=1, padding=1)

        self.ms_bias = ConvBNReLU(md_channels, md_channels, kernel_size=3, stride=1, padding=1)
        self.outconv = ConvBNReLU(md_channels, ms_out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
    def forward(self, Ms):


        ms_base = self.ms_cdc(Ms)
        ms_1 = self.ms_branch_1(ms_base)
        ms_2 = self.ms_branch_2(ms_base)
        ms_3 = self.ms_branch_3(ms_base)
        ms_detail = self.ms_detail_fuse(torch.cat([ms_1, ms_2, ms_3], dim=1))
        ms_clean = self.ms_detail_refine(ms_detail)   # 注意：这里不要外面再 + 一次
        ms_skip = self.ms_skip_proj(Ms)
        ms = torch.cat([ms_skip, ms_clean], dim=1)
        md_out = self.outconv(ms)

        # 保持原 decoder 的 cat 顺序：[浅层, 深层]
        return md_out
class BiDirectionalFusion_shen(nn.Module):
    """
    双向、非对称深浅融合块

    Md: 深层特征（已上采样到与 Ms 同分辨率）
    Ms: 浅层特征

    输出:
        torch.cat([ms_out, md_out], dim=1)

    参数:
        md_channels: 深层输入/输出通道
        ms_in_channels: 浅层输入通道
        ms_out_channels: 浅层输出通道
            - 若为 None，则默认与 ms_in_channels 相同
            - 用于处理 0_4 这种 Ms 输入 32ch，但最终只想输出 16ch 的情况
    """
    def __init__(self, md_channels, ms_in_channels, ms_out_channels=None, reduction=4, cdc_theta=0.7):
        super().__init__()
        if ms_out_channels is None:
            ms_out_channels = ms_in_channels

        self.md_channels = md_channels
        self.ms_in_channels = ms_in_channels
        self.ms_out_channels = ms_out_channels

        # ===== 1) 深层语义整理分支 =====
        self.md_se = SEBlock(md_channels, reduction=reduction)
        md_splits = _split_channels(md_channels, 3)
        self.md_branches = nn.ModuleList([
            DWPWBranch(md_channels, md_splits[0], kernel_size=3),
            DWPWBranch(md_channels, md_splits[1], kernel_size=5),
            DWPWBranch(md_channels, md_splits[2], kernel_size=7),
        ])
        self.md_fuse = ConvBNReLU(sum(md_splits), md_channels, kernel_size=1, padding=0)

        # ===== 2) 浅层细节净化分支 =====
        self.ms_cdc = CDCConv2d(ms_in_channels, ms_out_channels, kernel_size=3, stride=1, padding=1, theta=cdc_theta)
        self.ms_branch_1 = nn.Sequential(
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1),
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.ms_branch_2 = nn.Sequential(
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1),
            ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.ms_branch_3 = ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=3, padding=1)
        self.ms_detail_fuse = ConvBNReLU(ms_out_channels * 3, ms_out_channels, kernel_size=1, padding=0)
        self.ms_detail_refine = ChannelResidualMLP(ms_out_channels, reduction=reduction)

        # 原始浅层 skip 投影到 ms_out_channels，避免浅层信息被完全改写
        self.ms_skip_proj = ConvBNReLU(ms_in_channels, ms_out_channels, kernel_size=1, padding=0)

        # ===== 3) 深 -> 浅：成熟语义指导浅层 =====
        self.md_to_ms = ConvBNReLU(md_channels, ms_out_channels, kernel_size=1, padding=0)
        self.md_gate = nn.Sequential(
            nn.Conv2d(ms_out_channels, ms_out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.md_bias = ConvBNReLU(ms_out_channels, ms_out_channels, kernel_size=1, padding=0)

        # 浅层最终融合
        self.ms_out_fuse = ConvBNReLU(ms_out_channels * 3, ms_out_channels, kernel_size=1, padding=0)

        # ===== 4) 浅 -> 深：干净细节补偿深层 =====
        self.ms_to_md = ConvBNReLU(ms_out_channels, md_channels, kernel_size=1, padding=0)

        self.ms_bias = ConvBNReLU(md_channels, md_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
    def forward(self, Md, Ms):
        # -------------------------------------------------
        # A. 深层：先形成成熟语义 md_feat
        # -------------------------------------------------
        md_att = self.md_se(Md) * Md

        # -------------------------------------------------
        # B. 浅层：先形成干净细节 ms_clean
        # -------------------------------------------------
        ms_base = self.ms_cdc(Ms)
        ms_1 = self.ms_branch_1(ms_base)
        ms_2 = self.ms_branch_2(ms_base)
        ms_3 = self.ms_branch_3(ms_base)
        ms_detail = self.ms_detail_fuse(torch.cat([ms_1, ms_2, ms_3], dim=1))
        ms_clean = self.ms_detail_refine(ms_detail)   # 注意：这里不要外面再 + 一次
        ms_skip = self.ms_skip_proj(Ms)

        # -------------------------------------------------
        # C. 深 -> 浅：用成熟语义 md_feat 指导浅层
        # -------------------------------------------------
        md_proj = self.md_to_ms(md_att)
        md_gate = self.md_gate(md_proj)
        md_bias = self.md_bias(md_proj)

        # 非对称指导：语义对细节做“门控 + 偏置”
        # ms_guided = ms_clean * (1.0 + md_gate) + md_bias
        ms_guided = ms_clean * md_gate + md_bias
        ms_out = self.ms_out_fuse(torch.cat([ms_guided, ms_skip, md_proj], dim=1))

        # -------------------------------------------------
        # D. 浅 -> 深：用干净细节 ms_clean 补偿深层
        # -------------------------------------------------

        ms_proj = self.ms_to_md(ms_clean)
        ms_bias = self.ms_bias(ms_proj)
        md_out = self.relu(md_att + ms_bias)
        # 保持原 decoder 的 cat 顺序：[浅层, 深层]
        return torch.cat([ms_out, md_out], dim=1)
class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None


    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class MyNet(nn.Module):
    def __init__(self, num_classes=1,input_channels=1, block=Res_block):
        super(MyNet, self).__init__()
        # 最简单的直接定义
        num_blocks = [1, 1, 1, 1]
        nb_filter = [16, 32, 64, 128, 256]
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.Upsample(scale_factor=0.5, mode='nearest')

        self.up_4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.up_8 = nn.Upsample(scale_factor=8, mode='nearest')
        self.up_16 = nn.Upsample(scale_factor=16, mode='nearest')

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.enc30 = BiDirectionalFusion_enc(nb_filter[4], nb_filter[3], nb_filter[3])
        self.enc10 = BiDirectionalFusion_enc(nb_filter[3], nb_filter[2], nb_filter[2])
        self.enc00 = BiDirectionalFusion_enc(nb_filter[2], nb_filter[1], nb_filter[1])
        # self.enc00 = BiDirectionalFusion_enc(nb_filter[1], nb_filter[0], nb_filter[0])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3], num_blocks[2])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2], num_blocks[1])
        self.ca6 = ChannelAttention_max(nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1], num_blocks[0])
        self.ca7 = ChannelAttention_max(nb_filter[1])
        self.conv0_4 = self._make_layer(block, 64, nb_filter[0])
        self.ca8 = ChannelAttention_max(nb_filter[0])

        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        self.conv0_1 = self._make_layer(block, nb_filter[0], nb_filter[1])
        self.tox01_20 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.sup40 = nn.Conv2d(nb_filter[4], num_classes, kernel_size=1)
        self.sup31 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=1)
        self.sup22 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)
        self.sup13 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)
        self.sup04 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.fuse3_1 = BiDirectionalFusion(
            md_channels=nb_filter[4],  # 256
            ms_in_channels=nb_filter[3],  # 128
            ms_out_channels=nb_filter[3]  # 128
        )

        self.fuse2_2 = BiDirectionalFusion(
            md_channels=nb_filter[3],  # 128
            ms_in_channels=nb_filter[2],  # 64
            ms_out_channels=nb_filter[2]  # 64
        )

        self.fuse1_3 = BiDirectionalFusion(
            md_channels=nb_filter[2],  # 64
            ms_in_channels=nb_filter[1],  # 32
            ms_out_channels=nb_filter[1]  # 32
        )


        self.fuse0_4 = BiDirectionalFusion(
            md_channels=nb_filter[1],  # 32
            ms_in_channels=nb_filter[1],  # tox0_4 实际是 32
            ms_out_channels=nb_filter[0]  # 最终浅层输出 16
        )
        self.tox13 = nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=3, stride=2, padding=1)
        self.tox04 = nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=1, stride=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):

        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.enc00(x1_0) + x1_0
        x0_1 = self.conv0_1(x0_0)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.enc10(x2_0) + x2_0
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(self.fuse3_1(Md=self.up(x4_0), Ms=x3_0))
        x2_2 = self.conv2_2(self.fuse2_2(Md=self.up(x3_1), Ms=x2_0))
        x2_2 = self.ca6(x2_2) * x2_2
        x1_3 = self.conv1_3(self.fuse1_3(Md=self.up(x2_2), Ms=x1_0))
        x1_3 = self.ca7(x1_3) * x1_3

        x0_4 = self.conv0_4(self.fuse0_4(Md=self.up(x1_3), Ms=x0_1))
        x0_4 = self.ca8(x0_4) * x0_4

        out40 = self.sup40(x4_0).sigmoid()
        out31 = self.sup31(x3_1).sigmoid()
        out22 = self.sup22(x2_2).sigmoid()
        out13 = self.sup13(x1_3).sigmoid()
        out04 = self.sup04(x0_4).sigmoid()

        return out04, out13, out22, out31, out40



# ====== 测试脚本入口 ======
if __name__ == "__main__":
    from thop import profile, clever_format
    block = MyNet()
    x = torch.randn(1, 1, 256, 256)
    # 使用thop计算FLOPs
    flops, params = profile(block, inputs=(x,))
    flops_g, params_g = clever_format([flops, params], "%.3f")
    print(f"Parameters: {params_g}")
    print(f"FLOPs: {flops_g}")
