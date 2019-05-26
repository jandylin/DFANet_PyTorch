from model.backbone import backbone
from model.decoder import Decoder


class DFANet:

    def __init__(self):
        self.backbone1 = backbone()
        self.backbone1_up = nn.UpsamplingBilinear2d(scale_factor=4)

        self.decoder = Decoder()

    def forward(self, x):
        enc1_2, enc1_3, enc1_4, fc1, fca1 = self.backbone1(x)

