import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import get_norm


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn_norm, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            get_norm(bn_norm, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class MRBranch(nn.Module):
    """
    单路浅层卷积分支
    第一版建议：2层 3x3 Conv
    """
    def __init__(self, channels, bn_norm, num_layers=2, bias=False):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ConvBNAct(channels, channels, kernel_size=3, bn_norm=bn_norm, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class WeightNet(nn.Module):
    """
    三路动态权重生成模块
    输入：三路特征
    输出：alpha_short, alpha_mid, alpha_long
    """
    def __init__(self, channels, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(channels * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, f_s, f_m, f_l):
        # GAP: (B, C, F, T) -> (B, C)
        z_s = f_s.mean(dim=(-1, -2))
        z_m = f_m.mean(dim=(-1, -2))
        z_l = f_l.mean(dim=(-1, -2))

        z = torch.cat([z_s, z_m, z_l], dim=1)   # (B, 3C)
        z = F.relu(self.fc1(z))
        alpha = torch.softmax(self.fc2(z), dim=1)  # (B, 3)

        a_s = alpha[:, 0].view(-1, 1, 1, 1)
        a_m = alpha[:, 1].view(-1, 1, 1, 1)
        a_l = alpha[:, 2].view(-1, 1, 1, 1)
        return a_s, a_m, a_l


class MRFrontend(nn.Module):
    """
    多分辨率前端
    说明：
    - 中窗路不再单独做stem，直接使用DTT原始first_conv之后的特征f_base
    - 长窗、短窗各自做1x1stem
    - 然后统一对齐到中窗域
    - 三路浅层卷积分支
    - 动态权重融合
    """
    def __init__(
        self,
        dim_c_in,
        g,
        bn_norm,
        num_branch_layers=2,
        weight_hidden_dim=128,
        bias=False,
        align_mode="bilinear",
        fused_mode="weighted",
    ):
        super().__init__()

        self.align_mode = align_mode
        self.fused_mode = fused_mode

        # 只有 long / short 需要额外 stem
        self.stem_short = nn.Sequential(
            nn.Conv2d(dim_c_in, g, kernel_size=1, bias=bias),
            get_norm(bn_norm, g),
            nn.ReLU()
        )

        self.stem_long = nn.Sequential(
            nn.Conv2d(dim_c_in, g, kernel_size=1, bias=bias),
            get_norm(bn_norm, g),
            nn.ReLU()
        )

        # 三路浅层卷积分支：结构相同，参数独立
        self.branch_short = MRBranch(g, bn_norm, num_layers=num_branch_layers, bias=bias)
        self.branch_mid = MRBranch(g, bn_norm, num_layers=num_branch_layers, bias=bias)
        self.branch_long = MRBranch(g, bn_norm, num_layers=num_branch_layers, bias=bias)

        self.weight_net = WeightNet(g, hidden_dim=weight_hidden_dim)

    def _align_to_mid(self, x, target_hw):
        # x: (B, C, F, T)
        # target_hw: (F_mid, T_mid)
        return F.interpolate(x, size=target_hw, mode=self.align_mode, align_corners=False)

    def forward(self, x_short, f_mid_base, x_long):
        """
        参数：
        x_short: 短窗原始谱图，shape=(B, dim_c_in, F_s, T_s)
        f_mid_base: 中窗 first_conv 后特征，shape=(B, g, F_m, T_m)
        x_long: 长窗原始谱图，shape=(B, dim_c_in, F_l, T_l)

        返回：
        F_fused: shape=(B, g, F_m, T_m)
        """
        target_hw = f_mid_base.shape[-2:]

        f_s0 = self.stem_short(x_short)
        f_l0 = self.stem_long(x_long)

        f_s0 = self._align_to_mid(f_s0, target_hw)
        f_l0 = self._align_to_mid(f_l0, target_hw)

        f_s = self.branch_short(f_s0)
        f_m = self.branch_mid(f_mid_base)
        f_l = self.branch_long(f_l0)

        a_s, a_m, a_l = self.weight_net(f_s, f_m, f_l)

        if self.fused_mode == "long_only":
            f_fused = f_l
        else:
            f_fused = a_s * f_s + a_m * f_m + a_l * f_l

        return f_fused