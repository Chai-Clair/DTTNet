import torch.nn as nn
import torch

from src.dp_tdf.modules import TFC_TDF, TFC_TDF_Res1, TFC_TDF_Res2
from src.dp_tdf.bandsequence import BandSequenceModelModule

from src.layers import (get_norm)
from src.dp_tdf.abstract import AbstractModel
from src.dp_tdf.mr_frontend import MRFrontend

class DPTDFNet(AbstractModel):
    def __init__(
            self,
            num_blocks,
            l,
            g,
            k,
            bn,
            bias,
            bn_norm,
            bandsequence,
            block_type,
            mr_frontend=None,
            fusion_low_bins=None,
            **kwargs
    ):

        super(DPTDFNet, self).__init__(**kwargs)
        # self.save_hyperparameters()

        self.num_blocks = num_blocks #U-Net主框架encoder和decoder共有多少个block
        self.l = l #
        self.g = g  #通道数增量
        self.k = k  #应该是卷积核大小
        self.bn = bn
        self.bias = bias    #卷积层/线性层要不要带偏置项（bias）

        self.n = num_blocks // 2    #encoder和decoder各一半
        scale = (2, 2)  #上采样和下采样都按2倍缩放

        if block_type == "TFC_TDF":
            T_BLOCK = TFC_TDF
        elif block_type == "TFC_TDF_Res1":
            T_BLOCK = TFC_TDF_Res1
        elif block_type == "TFC_TDF_Res2":
            T_BLOCK = TFC_TDF_Res2
        else:
            raise ValueError(f"Unknown block type {block_type}")

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_c_in, out_channels=g, kernel_size=(1, 1)),
            get_norm(bn_norm, g),
            nn.ReLU(),
        )

        self.use_mr_frontend = mr_frontend is not None
        self.fusion_low_bins = fusion_low_bins

        if self.use_mr_frontend:
            self.mr_frontend = MRFrontend(
                dim_c_in=self.dim_c_in,
                g=g,
                bn_norm=bn_norm,
                bias=bias,
                **mr_frontend
            )
            # 静态可学习残差缩放系数 lambda
            self.fusion_scale = nn.Parameter(torch.tensor(0.1))

            # 固定频带 mask：shape = (1, 1, dim_f, 1)
            if self.fusion_low_bins is not None:
                assert self.fusion_low_bins > 0, "fusion_low_bins must be positive"
                assert self.fusion_low_bins <= self.dim_f, (
                    f"fusion_low_bins={self.fusion_low_bins} must be <= dim_f={self.dim_f}"
                )

                fusion_freq_mask = torch.zeros(1, 1, self.dim_f, 1)
                fusion_freq_mask[:, :, :self.fusion_low_bins, :] = 1.0
                self.register_buffer("fusion_freq_mask", fusion_freq_mask)
            else:
                self.fusion_freq_mask = None
        else:
            self.fusion_freq_mask = None

        f = self.dim_f  #当前频率大小，下采样f=f/2，上采样f=f*2
        c = g   #first_conv后的通道数
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()

        for i in range(self.n):
            c_in = c

            self.encoding_blocks.append(T_BLOCK(c_in, c, l, f, k, bn, bn_norm, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c + g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c + g),
                    nn.ReLU()
                )
            )
            f = f // 2
            c += g

        self.bottleneck_block1 = T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias)
        self.bottleneck_block2 = BandSequenceModelModule(
            **bandsequence,
            input_dim_size=c,
            hidden_dim_size=2*c
        )

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            # print(f"i: {i}, in channels: {c}")
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=c, out_channels=c - g, kernel_size=scale, stride=scale),
                    get_norm(bn_norm, c - g),
                    nn.ReLU()
                )
            )

            f = f * 2
            c -= g

            self.decoding_blocks.append(T_BLOCK(c, c, l, f, k, bn, bn_norm, bias=bias))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=self.dim_c_out, kernel_size=(1, 1)),
        )

    def forward(self, x):
        """
            两种输入形式：
            1. 原始 DTT：x是tensor，shape=(B, C_in, F, T)
            2. 严格版多窗前端：x是dict，包含short/mid/long
        """
        if isinstance(x, dict):
            x_short = x["short"]
            x_mid = x["mid"]
            x_long = x["long"]
            # 中窗主路：复用原始first_conv
            f_base = self.first_conv(x_mid)
            if self.use_mr_frontend:
                f_fused = self.mr_frontend(x_short, f_base, x_long)
                if self.fusion_freq_mask is not None:
                    f_fused = f_fused * self.fusion_freq_mask
                x = f_base + self.fusion_scale * f_fused
            else:
                x = f_base
        else:
            # 兼容原始单窗逻辑
            x = self.first_conv(x)

        x = x.transpose(-1, -2)
        ds_outputs = []
        for i in range(self.n):
            x = self.encoding_blocks[i](x)
            ds_outputs.append(x)
            x = self.ds[i](x)

        # print(f"bottleneck in: {x.shape}")
        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)

        for i in range(self.n):
            x = self.us[i](x)
            # print(f"us{i} in: {x.shape}")
            # print(f"ds{i} out: {ds_outputs[-i - 1].shape}")
            x = x * ds_outputs[-i - 1]
            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x