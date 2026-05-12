import torch.nn as nn
import torch

from src.dp_tdf.modules import TFC_TDF, TFC_TDF_Res1, TFC_TDF_Res2
from src.dp_tdf.bandsequence import BandSequenceModelModule

from src.layers import get_norm
from src.dp_tdf.abstract import AbstractModel
from src.dp_tdf.mr_frontend import MRFrontend


class DPTDFNet(AbstractModel):
    def __init__(self, num_blocks, l, g, k, bn, bias, bn_norm, bandsequence, block_type, mr_frontend=None, **kwargs):

        super(DPTDFNet, self).__init__(**kwargs)
        # self.save_hyperparameters()

        self.num_blocks = num_blocks  # U-Net主框架encoder和decoder共有多少个block
        self.l = l
        self.g = g  # 通道数增量
        self.k = k  # 应该是卷积核大小
        self.bn = bn
        self.bias = bias  # 卷积层/线性层要不要带偏置项（bias）

        self.n = num_blocks // 2  # encoder和decoder各一半
        scale = (2, 2)  # 上采样和下采样都按2倍缩放

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

        if self.use_mr_frontend:
            self.mr_frontend = MRFrontend(
                dim_c_in=self.dim_c_in,
                g=g,
                bn_norm=bn_norm,
                bias=bias,
                **mr_frontend
            )
            # 残差缩放系数，第一版用可学习标量
            self.fusion_scale = nn.Parameter(torch.tensor(0.1))

        f = self.dim_f  # 当前频率大小，下采样f=f/2，上采样f=f*2
        c = g  # first_conv后的通道数
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
            hidden_dim_size=2 * c
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

            skip = ds_outputs[-i - 1]

            # AMP(fp16) 下 decoder skip 乘法偶发溢出：
            # 只把 x * skip 这一处临时放到 fp32 计算；
            # 然后 clamp 到 fp16 可表示范围，再转回 fp16。
            # 这样不改变整体训练精度配置，也不把整个 decoder 改成 fp32。
            if x.dtype == torch.float16 or skip.dtype == torch.float16:
                x = (x.float() * skip.float()).clamp(
                    min=-torch.finfo(torch.float16).max,
                    max=torch.finfo(torch.float16).max,
                ).to(dtype=torch.float16)
            else:
                x = x * skip
            #x = x * ds_outputs[-i - 1]

            x = self.decoding_blocks[i](x)

        x = x.transpose(-1, -2)

        x = self.final_conv(x)

        return x