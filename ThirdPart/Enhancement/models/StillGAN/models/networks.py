# -*- coding: utf-8 -*-

import functools
import torch
import torch.nn as nn


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class res_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer):
        super(res_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(ch_out),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=False),
            norm_layer(ch_out),
        )
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv(x)

        return self.relu(out + residual)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer, use_dropout=False):
        super(up_conv,self).__init__()
        if use_dropout:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.5)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ch_out),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = self.up(x)

        return x


class ResUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(2)

        self.Conv1 = res_conv_block(ch_in=img_ch, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]
        self.Conv2 = res_conv_block(ch_in=ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]
        self.Conv3 = res_conv_block(ch_in=2 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]
        self.Conv4 = res_conv_block(ch_in=4 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 8, W / 8]
        self.Conv5 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 16, W / 16]
        self.Conv6 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 32, W / 32]
        self.Conv7 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 64, W / 64]
        self.Conv8 = res_conv_block(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 128, W / 128]

        self.Up8 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 128, W / 128]
        self.Up_conv8 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 128, W / 128]

        self.Up7 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 64, W / 64]
        self.Up_conv7 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 64, W / 64]

        self.Up6 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 32, W / 32]
        self.Up_conv6 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 32, W / 32]

        self.Up5 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 16, W / 16]
        self.Up_conv5 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 16, W / 16]

        self.Up4 = up_conv(ch_in=8 * ngf, ch_out=8 * ngf, norm_layer=norm_layer, use_dropout=use_dropout)  # [B, 8 * ngf, H / 8, W / 8]
        self.Up_conv4 = res_conv_block(ch_in=16 * ngf, ch_out=8 * ngf, norm_layer=norm_layer)  # [B, 8 * ngf, H / 8, W / 8]

        self.Up3 = up_conv(ch_in=8 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]
        self.Up_conv3 = res_conv_block(ch_in=8 * ngf, ch_out=4 * ngf, norm_layer=norm_layer)  # [B, 4 * ngf, H / 4, W / 4]

        self.Up2 = up_conv(ch_in=4 * ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]
        self.Up_conv2 = res_conv_block(ch_in=4 * ngf, ch_out=2 * ngf, norm_layer=norm_layer)  # [B, 2 * ngf, H / 2, W / 2]

        self.Up1 = up_conv(ch_in=2 * ngf, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]
        self.Up_conv1 = res_conv_block(ch_in=2 * ngf, ch_out=ngf, norm_layer=norm_layer)  # [B, ngf, H, W]

        self.Conv_1x1 = nn.Conv2d(ngf, output_ch, kernel_size=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # [B, ngf, H, W]

        x2 = self.Maxpool(x1)  # [B, ngf, H / 2, W / 2]
        x2 = self.Conv2(x2)  # [B, 2 * ngf, H / 2, W / 2]

        x3 = self.Maxpool(x2)  # [B, 2 * ngf, H / 4, W / 4]
        x3 = self.Conv3(x3)  # [B, 4 * ngf, H / 4, W / 4]

        x4 = self.Maxpool(x3)  # [B, 4 * ngf, H / 8, W / 8]
        x4 = self.Conv4(x4)  # [B, 8 * ngf, H / 8, W / 8]

        x5 = self.Maxpool(x4)  # [B, 8 * ngf, H / 16, W / 16]
        x5 = self.Conv5(x5)  # [B, 8 * ngf, H / 16, W / 16]

        x6 = self.Maxpool(x5)  # [B, 8 * ngf, H / 32, W / 32]
        x6 = self.Conv6(x6)  # [B, 8 * ngf, H / 32, W / 32]

        x7 = self.Maxpool(x6)  # [B, 8 * ngf, H / 64, W / 64]
        x7 = self.Conv7(x7)  # [B, 8 * ngf, H / 64, W / 64]

        x8 = self.Maxpool(x7)  # [B, 8 * ngf, H / 128, W / 128]
        x8 = self.Conv8(x8)  # [B, 8 * ngf, H / 128, W / 128]

        x9 = self.Maxpool(x8)  # [B, 8 * ngf, H / 256, W / 256]

        # decoding + concat path
        d8 = self.Up8(x9)  # [B, 8 * ngf, H / 128, W / 128]
        d8 = torch.cat((x8, d8), dim=1)  # [B, 16 * ngf, H / 128, W / 128]
        d8 = self.Up_conv8(d8)  # [B, 8 * ngf, H / 128, W / 128]

        d7 = self.Up7(d8)  # [B, 8 * ngf, H / 64, W / 64]
        d7 = torch.cat((x7, d7), dim=1)  # [B, 16 * ngf, H / 64, W / 64]
        d7 = self.Up_conv7(d7)  # [B, 8 * ngf, H / 64, W / 64]

        d6 = self.Up6(d7)  # [B, 8 * ngf, H / 32, W / 32]
        d6 = torch.cat((x6, d6), dim=1)  # [B, 16 * ngf, H / 32, W / 32]
        d6 = self.Up_conv6(d6)  # [B, 8 * ngf, H / 32, W / 32]

        d5 = self.Up5(d6)  # [B, 8 * ngf, H / 16, W / 16]
        d5 = torch.cat((x5, d5), dim=1)  # [B, 16 * ngf, H / 16, W / 16]
        d5 = self.Up_conv5(d5)  # [B, 8 * ngf, H / 16, W / 16]

        d4 = self.Up4(d5)  # [B, 8 * ngf, H / 8, W / 8]
        d4 = torch.cat((x4, d4), dim=1)  # [B, 16 * ngf, H / 8, W / 8]
        d4 = self.Up_conv4(d4)  # [B, 8 * ngf, H / 8, W / 8]

        d3 = self.Up3(d4)  # [B, 4 * ngf, H / 4, W / 4]
        d3 = torch.cat((x3, d3), dim=1)  # [B, 8 * ngf, H / 4, W / 4]
        d3 = self.Up_conv3(d3)  # [B, 4 * ngf, H / 4, W / 4]

        d2 = self.Up2(d3)  # [B, 2 * ngf, H / 2, W / 2]
        d2 = torch.cat((x2, d2), dim=1)  # [B, 4 * ngf, H / 2, W / 2]
        d2 = self.Up_conv2(d2)  # [B, 2 * ngf, H / 2, W / 2]

        d1 = self.Up1(d2)  # [B, ngf, H, W]
        d1 = torch.cat((x1, d1), dim=1)  # [B, 2 * ngf, H, W]
        d1 = self.Up_conv1(d1)  # [B, ngf, H, W]

        out = nn.Tanh()(self.Conv_1x1(d1))

        return out
