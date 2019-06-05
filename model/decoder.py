import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):
        self.enc1_conv = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3)  # not sure about the out channels
        #self.enc1_up = nn.UpsamplingBilinear2d(scale_factor=1)  # no upsampling required?

        self.enc2_up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.enc2_conv = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3)

        self.enc3_conv = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3)
        self.enc3_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.enc_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        #self.enc_up = nn.UpsamplingBilinear2d(scale_factor=1)  # again..

        self.fca1_conv = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=3)
        self.fca1_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.fca2_conv = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=3)
        self.fca2_up = nn.UpsamplingBilinear2d(scale_factor=8)

        self.fca3_conv = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=3)
        self.fca3_up = nn.UpsamplingBilinear2d(scale_factor=16)

        self.final_up = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, enc1, enc2, enc3, fca1, fca2, fca3):
        """Note that enc1 denotes the output of the enc4 module of backbone instance 1."""
        #e1 = self.enc1_up(self.enc1_conv(enc1))
        e1 = self.enc1_conv(enc1)
        e2 = self.enc2_up(self.enc2_conv(enc2))
        e3 = self.enc3_up(self.enc3_conv(enc3))
        #e = self.enc_up(self.enc_conv(e1 + e2 + e3))
        e = self.enc_conv(e1 + e2 + e3)
        f1 = self.fca1_up(self.fca1_conv(fca1))
        f2 = self.fca2_up(self.fca1_conv(fca2))
        f3 = self.fca3_up(self.fca1_conv(fca3))
        o = self.final_up(e + f1 + f2 + f3 + o)
        return o

