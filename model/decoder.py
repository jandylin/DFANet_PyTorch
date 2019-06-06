import math
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_relu = self.relu(x)
        x_conv = self.conv(x_relu)
        x_bn = self.bn(x_conv)
        return x_bn


class Decoder(nn.Module):

    def __init__(self, n_classes=19):
        super(Decoder, self).__init__()
        self.n_classes = n_classes
        self.enc1_conv = ConvBlock(48, 32, 1) # not sure about the out channels

        self.enc2_conv = ConvBlock(48, 32, 1)
        self.enc2_up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc3_conv = ConvBlock(48, 32, 1)
        self.enc3_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.enc_conv = ConvBlock(32, n_classes, 1)

        self.fca1_conv = ConvBlock(192, n_classes, 1)
        self.fca1_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.fca2_conv = ConvBlock(192, n_classes, 1)
        self.fca2_up = nn.UpsamplingBilinear2d(scale_factor=8)

        self.fca3_conv = ConvBlock(192, n_classes, 1)
        self.fca3_up = nn.UpsamplingBilinear2d(scale_factor=16)

        self.final_up = nn.UpsamplingBilinear2d(scale_factor=4)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, enc1, enc2, enc3, fca1, fca2, fca3):
        """Note that enc1 denotes the output of the enc4 module of backbone instance 1."""
        e1 = self.enc1_conv(enc1)
        e2 = self.enc2_up(self.enc2_conv(enc2))
        e3 = self.enc3_up(self.enc3_conv(enc3))

        e = self.enc_conv(e1 + e2 + e3)

        f1 = self.fca1_up(self.fca1_conv(fca1))
        f2 = self.fca2_up(self.fca1_conv(fca2))
        f3 = self.fca3_up(self.fca1_conv(fca3))

        o = self.final_up(e + f1 + f2 + f3)

        return o

